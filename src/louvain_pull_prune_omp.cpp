/*
* Copyright (C) 2020 Intel Corporation
*
* Redistribution and use in source and binary forms, with or without modification,
* are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
*    this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
* 3. Neither the name of the copyright holder nor the names of its contributors
*    may be used to endorse or promote products derived from this software
*    without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
* THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
* BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
* OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
* OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
* OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
* WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
* OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
* EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* SPDX-License-Identifier: BSD-3-Clause
*/


#include "louvain_pull_prune_omp.h"


/********************************** HASHing function *******************************/
// near power of 2 for u64
inline u64 near_power_of_2(u64 v) {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v |= v >> 32;
	v++;
	return v;
}

static inline u64 compute_max_hmap_size(u64 deg) {
	u64 l2;
	u64 hmsize = deg;
	for (l2 = 1; hmsize > (1 << l2); l2++)
		;
	hmsize = (1 << (l2)) - 1;
	return hmsize;
}

static inline u32 compute_logsize(u64 max_entries) {
	u32 l2;
	for (l2 = 1; max_entries > (1 << l2); l2++)
		;
	return l2;
}

static inline u64 myHASH(u64 x) {
	u64 s = 0x9e3779b97f4a7c15;
	x *= s;
	return x;
}

static inline u64 HASH(u64 x, u64 lsize) {

	return (x * 0x9e3779b97f4a7c15) & (lsize);
}

static inline u32 HASH_insert(const u64 ck, neigh_comm_map_t* hmap,
	const double edge_weight, const u64 hmsize, const u64 lsize,
	const u64 max_key,
	u64* neigh_pos, u32* num_comm) {
	u32 nc = 0;
	u64 l = HASH(ck, lsize);
	for (l = l & hmsize; // init
		hmap[l].comm_id != ck && hmap[l].comm_id != max_key; // test
		l = ((l + 1) & hmsize), nc++) {
	} // increment

	if (hmap[l].comm_id == ck) {
		hmap[l].weight_sum += edge_weight;
	}
	else if (hmap[l].comm_id == max_key) {
		hmap[l].comm_id = ck;
		hmap[l].weight_sum = edge_weight;
		neigh_pos[*num_comm] = l;
		*num_comm = *num_comm + 1;
	}
	else
		printf("HASH_insert error: ck=%llu, l=%llu hmap[l].comm_id=%llu\n", ck, l, hmap[l].comm_id);

	return nc;
}

/****************************************************************************************************/

/********************************************* graph prepossing *************************************/
// Function removes duplicate elements from a sorted array and
// returns new size of modified array.
u64 removeDuplicates(neighbor_t* adj, u64 length) {
	if (length == 0 || length == 1)
		return length;
	
	u64 j = 0;
	neighbor_t* temp = (neighbor_t*)malloc(sizeof(neighbor_t) * length);

	if(NULL == temp){
		printf("[removeDuplicates] <temp> allocation ERROR. Returning 0.");
		return j;
	}

	// Start traversing elements

	for (u64 i = 0; i < length - 1; i++) {
		if (adj[i].neigh_id != adj[i + 1].neigh_id)
			temp[j++] = adj[i];
	}

	temp[j++] = adj[length - 1];

#ifdef SORT_BY_DEGREE
	std::sort(temp, temp + j, sortByDeg);
#endif

#ifdef REMOVE_ZERO_DEGREE_NEIGBORS
	for (u64 i = j - 1; i >= 0; i--)
	{
		if (G[temp[i]].count > 0)
		{
			j = i + 1;
			break;
		}
	}
#endif
	// Modify original array
#pragma omp parallel for
	for (u64 i = 0; i < j; i++)
		adj[i] = temp[i];

	free(temp);
	return j;
}

/* Free Memory */
void deallocate_memory() {
	free(n2c);
	free(community);

	for (int i = 0; i < p; i++) {
		free(thread_HASHmap[i]);
		free(thread_HASH_nonempty_loc[i]);
	}
	free(thread_HASHmap);
	free(thread_HASH_nonempty_loc);
#ifdef PRUNE
	free(recompute);
#endif
}

/* initialize communities and other data structures */
void louvain_init_community(graph_t* g, const u64 n, community_t* community,
	const int num_thread) {
	double my_weight[num_thread];
	u64 mdeg[num_thread];
	for (int t = 0; t < num_thread; t++) {
		my_weight[t] = 0.;
		mdeg[t] = 0;
	}
#pragma omp for schedule(static,1)
	for (int t = 0; t < num_chunk; t++) {
		int thread_id = omp_get_thread_num();
		u64 start = t * CHUNK_SIZE;
		u64 end = (start + CHUNK_SIZE <= n) ? (start + CHUNK_SIZE) : n;

		//printf("start chunk %d end chunk %d\n", start, end);
		for (int i = start; i < end; i++) {
			n2c[i] = i;
			u64 deg = g[i].deg;
			if (deg == 0) {
				community[i].tot = 0.;
				community[i].in = 0.;
#ifdef PRUNE
				recompute[i] = 0;
#endif
				continue;
			}
			neighbor_t* edge_list = g[i].offset;
#ifdef PRUNE
			recompute[i] = 1;
#endif
			double edge_weight = 0.0; //w(u)
			double self_weight = 0.0;

			for (int j = 0; j < deg; j++) {
				edge_weight += edge_list[j].edge_weight;
				if (self_weight == 0. && edge_list[j].neigh_id == i)
					self_weight = edge_list[j].edge_weight;
			}
			g[i].total_edge_weight = edge_weight;
			g[i].self_loop_weight = self_weight;
			//printf("vertex i %d with weight %f \n", i, edge_weight);
			community[i].tot = edge_weight;
			community[i].in = self_weight;
			my_weight[thread_id] += edge_weight;
			//printf("thread t %d with weight %f \n", thread_id, my_weight[thread_id]);
			if (mdeg[thread_id] < deg)
				mdeg[thread_id] = deg;
		}
	}

	for (int t = 0; t < num_thread; t++) {
		total_graph_edge_weight += my_weight[t];
		max_deg = MAX(mdeg[t], max_deg);
	}

}

/* Phase 1 (inner loop) of Louvain */
void louvain_inner_loop(const u64 htblsize, const u64 maxhmsize, const u64 max_key, const int num_thread,
	community_t* community, graph_t* g, const u64 lsize, neigh_comm_map_t** thread_HASHmap, u64** thread_HASH_nonempty_loc,
	omp_lock_t* restrict comm_lock, omp_lock_t* restrict vertex_lock) {
	//1. compute initial modularity
	double in_sum_c = 0;
	double tot_sum_c = 0;
	const u64 N = n;
#pragma omp parallel for reduction(+:tot_sum_c) reduction(+:in_sum_c)
	for (u64 i = 0; i < N; i++)
	{
#if 0
		cout << "i " << i << " :" << community[i].in << "," << community[i].tot << endl;
#endif
		if (community[i].tot > 0.0)
		{

			double tm2 = (community[i].tot);
			tot_sum_c += (tm2 * tm2);
			in_sum_c += community[i].in;
		}
	}
	modularity = (in_sum_c - tot_sum_c * one_over_m2) * one_over_m2;
	printf("Initial modularity %f\n", modularity);


	/*********************************************************************/
	double cur_mod;
	int iter = 0;

	// 2. initialize thread private HASH table
#pragma omp parallel for //proc_bind(close)
	for (int t = 0; t < num_thread; t++) {
		int thread_id = omp_get_thread_num();
		neigh_comm_map_t* neigh_comm_map = thread_HASHmap[thread_id];
#pragma vector
		for (u64 i = 0; i < htblsize; i++) {
			neigh_comm_map[i].comm_id = max_key;
		}

	}

	// loop until modularity increase by a threshold
	do {
		cur_mod = modularity;
		// for each vertex chunk in parallel
#pragma omp parallel for schedule(static,1)
		for (int t = 0; t < num_chunk; t++)
		{
			int thread_id = omp_get_thread_num();
			neigh_comm_map_t* neigh_comm_map = thread_HASHmap[thread_id];
			u64* neigh_pos = thread_HASH_nonempty_loc[thread_id];

			u64 start = t * CHUNK_SIZE;
			u64 end = (start + CHUNK_SIZE <= n) ? (start + CHUNK_SIZE) : n;

			// for each vertex in the chunk
			for (u64 i = start; i < end; i++) {
				const u64 deg = g[i].deg;
				if (deg == 0) continue;
#ifdef PRUNE
				if (recompute[i] == 0) continue;
#endif

				// compute NCW
				u32 num_comm = 0;
				u64 node_comm = n2c[i];
				u64 l = HASH(node_comm, lsize);
				l = l & maxhmsize;
				neigh_comm_map[l].comm_id = node_comm;
				neigh_comm_map[l].weight_sum = 0.;
				neigh_pos[num_comm] = l;
				num_comm = num_comm + 1;

				neighbor_t* edge_list = g[i].offset;

				for (u64 e = 0; e < deg; e++)
				{
					u64 neigh = edge_list[e].neigh_id;
					_mm_prefetch((char*)(edge_list + e + 64), _MM_HINT_T0);
					if (neigh != i)
					{

						u64 neigh_comm = n2c[neigh];

						HASH_insert(neigh_comm, neigh_comm_map, edge_list[e].edge_weight, maxhmsize, lsize, max_key, neigh_pos, &num_comm);

					}
				}

				// find the best community to move into
				u64 best_comm = node_comm;
				double best_increase = 0.;
				double best_eweight = 0.;

				double n_in_weight = g[i].total_edge_weight;
				double w_n_over_m2 = n_in_weight * one_over_m2;
				double neigh_weight_node_comm = neigh_comm_map[neigh_pos[0]].weight_sum;


#ifdef ANALYZE
				num_comm_total += num_comm;
#endif

				for (u32 hl = 0; hl < num_comm; hl++) {
					u64 HASH_loc = neigh_pos[hl];

					u64 comm_id = neigh_comm_map[HASH_loc].comm_id;
					double wn2c = neigh_comm_map[HASH_loc].weight_sum;

					//omp_set_lock(comm_lock + comm_id);
					double totc = community[comm_id].tot;
					//omp_unset_lock(comm_lock + comm_id);

					double increase = (wn2c - totc * w_n_over_m2);

					if (increase > best_increase) {
						best_comm = comm_id;
						best_eweight = wn2c;
						best_increase = increase;
					}
					neigh_comm_map[HASH_loc].comm_id = max_key;
				}

				// if best community is not the node_community, move
				if (best_comm != node_comm) {
#ifdef PRUNE
					recompute[i] = 1;
#endif
					double loops_tmp = g[i].self_loop_weight;

					omp_set_lock(comm_lock + node_comm);
					community[node_comm].tot -= n_in_weight;
					community[node_comm].in -= (2.0 * neigh_weight_node_comm + loops_tmp);
					omp_unset_lock(comm_lock + node_comm);

					omp_set_lock(comm_lock + best_comm);
					community[best_comm].tot += n_in_weight;
					community[best_comm].in += (2.0 * best_eweight + loops_tmp);
					omp_unset_lock(comm_lock + best_comm);

					// lock here is not necessary, same as Grappolo
					//omp_set_lock(vertex_lock + i);
					n2c[i] = best_comm;
					//omp_unset_lock(vertex_lock + i);
#ifdef ANALYZE
					num_moves_total += 1;
#endif
					// if pruning is enabled, turn recomputation on for the neighbors
#ifdef PRUNE
					for (u64 e = 0; e < deg; e++)
					{
#ifdef ANALYZE
						edges_explored++;
#endif
						u64 neigh = edge_list[e].neigh_id;

						if (n2c[neigh] != best_comm)
							recompute[neigh] = 1;
					}
#endif
				}
				// if there was no move, then do not recompute
#ifdef PRUNE
				else {
					recompute[i] = 0;
				}
#endif
			} //end vertex loop

		} //end of thread chunk loop
#ifdef ANALYZE
		pull_time += getMilliSpan(start_time);
		double time_span = pull_time - last_time;
		printf("pull based: edges explored at iter %d %llu in time %f \n", iter, edges_explored - edges_explored_last, time_span);
		edges_explored_last = edges_explored;
		last_time = pull_time;
#endif

		// recompute the modularity
		double in_sum_c = 0;
		double tot_sum_c = 0;
#pragma omp parallel for reduction(+:tot_sum_c) reduction(+:in_sum_c)
		for (u64 i = 0; i < n; i++)
		{
#if 0
			cout << "i " << i << " :" << community[i].in << "," << community[i].tot << endl;
#endif
			if (community[i].tot > 0.0)
			{

				double tm2 = (community[i].tot);
				//#pragma omp atomic
				tot_sum_c += (tm2 * tm2);
				//#pragma omp atomic
				in_sum_c += community[i].in;
			}
		}
		modularity = (in_sum_c - tot_sum_c * one_over_m2) * one_over_m2;
		iter++;

		printf("iteration %d new modularity %f\n", iter, modularity);
		//if(iter==2) break;
        

	} while (modularity - cur_mod > min_modularity );
}

int graph_reconstruction() {
	// Renumber communities
	u64* renumber = (u64*)malloc(sizeof(u64) * (n + 1));

	if(NULL == renumber) {
		printf("[graph_reconstruction] <RENUMBER> allocation ERROR. Exiting.\n");
		return 1;
	}

#pragma omp parallel for
	for (u64 i = 0; i < n + 1; i++)
		renumber[i] = 0;

	//count number of vertices in each community
	//#pragma omp parallel for
	for (u64 node = 0; node < n + 1; node++) {
		renumber[n2c[node]]++;
	}

	// renumber vertices
	u64 next_comm = 0;
	for (u64 i = 0; i < n + 1; i++) {
		if (renumber[i] == 0)
			continue;
		else
			renumber[i] = next_comm++;
	}

	// Compute communities
	u64 n2 = next_comm;
	u64 m2 = 0;

	// bin vertices in their respective communities
	vector < vector<u64> > comm_nodes(n2 + 1);
	for (u64 node = 0; node < n + 1; node++) {
		comm_nodes[renumber[n2c[node]]].push_back(node);
		//printf("%llu %llu\n", node, renumber[n2c[node]]);
	}

	// Compute weighted graph
	graph_t* G2;
	posix_memalign((void**)&G2, ALN, (n2 + 1) * sizeof(graph_t));
	assert(G2 != NULL);

	//initialize all degrees to zero
#pragma omp parallel for 
	for (u64 i = 0; i < n2 + 1; i++) {
		G2[i].deg = 0;
		G2[i].total_edge_weight = 0.;
		G2[i].self_loop_weight = 0.;
	}

	vector<neighbor_t> adjMat2;
	G2[0].offset = adjMat;
	total_graph_edge_weight = 0.;
	for (u64 comm = 0; comm < n2; comm++) {
		map<u64, double> edge_map;
		map<u64, double>::iterator it;

		u64 comm_size = comm_nodes[comm].size();
		for (u64 node = 0; node < comm_size; node++) {
			const u64 deg = G[comm_nodes[comm][node]].deg;
			neighbor_t* edge_list = G[comm_nodes[comm][node]].offset;
			for (u64 i = 0; i < deg; i++) {
				u64 neigh = edge_list[i].neigh_id;
				u64 neigh_comm = renumber[n2c[neigh]];
				double neigh_weight = edge_list[i].edge_weight;

				it = edge_map.find(neigh_comm);
				if (it == edge_map.end())
					edge_map.insert(make_pair(neigh_comm, neigh_weight));
				else
					it->second += neigh_weight;
			}
		}
		G2[comm].deg = edge_map.size();
		G2[comm + 1].offset = (neighbor_t*)(G2[comm].offset + edge_map.size());

		m2 += edge_map.size();

		for (it = edge_map.begin(); it != edge_map.end(); it++) {
			neighbor_t edge;
			edge.neigh_id = it->first;
			edge.edge_weight = it->second;
			adjMat2.push_back(edge);
			G2[comm].total_edge_weight += edge.edge_weight;
			total_graph_edge_weight += edge.edge_weight;
			if (edge.neigh_id == comm)
				G2[comm].self_loop_weight += edge.edge_weight;
		}
	}

	n = n2;
	m = m2;
	if(total_graph_edge_weight) one_over_m2 = 1.0 / total_graph_edge_weight;

#pragma omp parallel for
	for (u64 i = 0; i < n + 1; i++) {
		n2c[i] = i;
		G[i] = G2[i];
	}
	free(G2);
	free(renumber);
#pragma omp parallel for
	for (u64 i = 0; i < m; i++) {
		adjMat[i] = adjMat2[i];
	}

	num_chunk = (n + p - 1) / p;
	//printf("%llu %llu %d %d\n ", n, m, p, num_chunk);
#pragma omp for schedule(static,1)
	for (int t = 0; t < num_chunk; t++) {
		u64 start = t * CHUNK_SIZE;
		u64 end = (start + CHUNK_SIZE <= n) ? (start + CHUNK_SIZE) : n;

		//printf("start chunk %d end chunk %d\n", start, end);
		for (u64 i = start; i < end; i++) {
			n2c[i] = i;
			u64 deg = G[i].deg;
			if (deg == 0) {
				community[i].tot = 0.;
				community[i].in = 0.;
#ifdef PRUNE
				recompute[i] = 0;
#endif
				continue;
			}

#ifdef PRUNE
			recompute[i] = 1;
#endif

			community[i].tot = G[i].total_edge_weight;
			community[i].in = G[i].self_loop_weight;
		}

	}
	//printf("initialized communities\n");
   return 0;
}
int main(int argc, char** argv) {
	ofstream outfile;
	ifstream infile;
	const char* outpv;
	const char* inpv;
	int verbose = 0;
	int verify = 1;

	//help message
	if (argc > 1 && !strcasecmp(argv[1], "-h")) {
		printf("To run Louvain:\n./louvain -i input_file_in_txt -p threads");
		printf(
			" -o output_file (optional) -v (0 or 1 for verbose output) -verify (0 or 1)\n");
		printf("optional inputs: -v -p -o -verify\n");
		exit(0);
	}
	// take input file
	inpv = getParamVal(argc, argv, (char*)"-i");
	if (inpv == NULL) {
		inpv = "/nfs/mmdc/disks/tpi4/proj/USA-road-d.USA.txt";
	}
	infile.open(inpv);
	printf("Reading from file %s\n", inpv);
	if (!infile.is_open()) {
		cout << "Could not open the file:" << inpv << " exiting\n";
		exit(0);
	}
	// take output file
	outpv = getParamVal(argc, argv, (char*)"-o");
	if (outpv != NULL) {
		outfile.open(outpv);
	}
	p = 1;
	inpv = getParamVal(argc, argv, (char*)"-p");
#if 1
	if (inpv != NULL) {
		int p = atoi(inpv);

		p = omp_get_max_threads();

	}
	omp_set_dynamic(0);     // Explicitly disable dynamic teams

#pragma omp parallel
	{
		omp_set_num_threads(p);
		p = omp_get_num_threads();
	}

#endif
	inpv = getParamVal(argc, argv, (char*)"-v");
	if (inpv != NULL) {
		verbose = atoi(inpv);
		cout << "verbose mode on\n";
	}
	inpv = getParamVal(argc, argv, (char*)"-verify");
	if (inpv != NULL) {
		verify = atoi(inpv);
	}
	infile >> n;
	infile >> n;
	infile >> m;
	cout << "Running: " << argv[0] << ", N:" << n << ", M:" << m << ", threads:"
		<< p << endl;
	n = n + 1;
	num_chunk = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;

	if (num_chunk < p) {
		p = num_chunk;
#pragma omp parallel
		omp_set_num_threads(p);
	}
#pragma omp barrier

	printf("num_threads %d, num_chunk %d\n", p, num_chunk);
	//allocate memory for the graph
	posix_memalign((void**)&G, ALN, (n) * sizeof(graph_t));
	assert(G != NULL);
	u64* ofs;
	edge_t* edgeList;
	assert(ofs = new u64[n]);
	assert(edgeList = new edge_t[m]);

	//initialize all degrees to zero
#pragma omp parallel for 
	for (u64 i = 0; i < n; i++) {
		G[i].deg = 0;
		G[i].total_edge_weight = 0.;
		G[i].self_loop_weight = 0.;
	}
	//take edges from file and initialize the graph and also compute the degree of each vertex
	u64 i = 0;
	u64 expected_edges = m;
	u64 u, v;
	double w = 1.0;
	while (!infile.eof() && i < expected_edges) {
		infile >> u;
		infile >> v;

#ifdef WEIGHTED
		infile >> w;
#else
		w = 1.0;
#endif
		if (u == v)
			continue;
		edgeList[i].u = u;
		edgeList[i].v = v;
		edgeList[i].edge_weight = w;
		G[u].deg++;

		// reverse edge
#ifdef REVERSE_EDGE
		G[v].deg++;
#endif //REVERSE_EDGE
		i++;
	}
	infile.close();
	expected_edges = i;
	//printf("%llu %llu %llu %f\n", i, edgeList[i-1].u, edgeList[i-1].v, edgeList[i].edge_weight);
#ifdef REVERSE_EDGE
	m = i * 2;
#else
	m = i;
#endif

	u64 exs = ALN / sizeof(neighbor_t);
	//Pad adjMat to be a multiple of ALN factor
	posix_memalign((void**)&adjMat, ALN,
		(m + exs * n + 1) * sizeof(neighbor_t));
	if (adjMat == NULL) {
		cout << "Failed in adjMat allocation\n";
		exit(1);
	}
	//find the offset in adjMat for each vertex. offset basically stores the starting position of adjacency list of each vertex i, in the adjacency list.
	ofs[0] = 0;
	G[0].offset = (neighbor_t*)(adjMat + ofs[0]);
	for (u64 i = 1; i < n; i++) {
		ofs[i] = ofs[i - 1] + G[i - 1].deg;
		u64 pad = (G[i - 1].deg) % exs; //how much you need to pad to be a multiple of alignment factor
		if (pad)
			ofs[i] += (exs - pad);
		G[i].offset = (neighbor_t*)(adjMat + ofs[i]);
	}
	//#pragma parallel
	for (u64 i = 0; i < expected_edges; i++) {
		u64 j = edgeList[i].u;
		u64 k = ofs[j];
		adjMat[k].neigh_id = edgeList[i].v;
		adjMat[k].edge_weight = (double)edgeList[i].edge_weight;
		ofs[j] = k + 1;
#ifdef REVERSE_EDGE
		//reverse graph
		u64 j2 = edgeList[i].v;
		u64 k2 = ofs[j2];
		adjMat[k2].neigh_id = edgeList[i].u;
		adjMat[k2].edge_weight = (double)edgeList[i].edge_weight;
		ofs[j2] = k2 + 1;
#endif //REVERSE_EDGE
	}

	u64 actual_edges = 0;
#pragma omp parallel for
	for (u64 i = 0; i < n; i++) {
		if (G[i].deg > 1) {
			u64 k = ofs[i];
			// should use a parallel
			// sorting algorithm
#ifdef SORT_EDGES
			std::sort(adjMat + k - G[i].deg, adjMat + k);
#endif
#ifdef DEDUPLICATE
			G[i].deg = removeDuplicates(adjMat + k - G[i].deg, G[i].deg);
#endif //DEDUPLICATE
		}
#pragma omp atomic
		actual_edges += G[i].deg;
	}

	printf("actual edges %llu \n", actual_edges);
	delete[] edgeList;
	delete[] ofs;

#ifdef PRUNE
	posix_memalign((void**)&recompute, ALN, (n) * sizeof(u8));
#endif
	posix_memalign((void**)&n2c, ALN, (n) * sizeof(u64));
	assert(n2c != NULL);

	posix_memalign((void**)&community, ALN, (n) * sizeof(community_t));
	assert(community != NULL);

	omp_lock_t* restrict
		comm_lock = (omp_lock_t*)_mm_malloc(sizeof(omp_lock_t) * n, ALN);
	assert(comm_lock != NULL);

	omp_lock_t* restrict
		vertex_lock = (omp_lock_t*)_mm_malloc(sizeof(omp_lock_t) * n, ALN);
	assert(vertex_lock != NULL);

#pragma omp parallel for schedule (static, CHUNK_SIZE) 
	for (u64 i = 0; i < n; i++) {
		omp_init_lock(comm_lock + i);
		omp_init_lock(vertex_lock + i);
	}

	louvain_init_community(G, n, community, p);
	thread_HASHmap = (neigh_comm_map_t**)malloc(
		(p) * sizeof(neigh_comm_map_t*));
	assert(thread_HASHmap != NULL);

	thread_HASH_nonempty_loc = (u64**)malloc((p) * sizeof(u64*));
	assert(thread_HASH_nonempty_loc != NULL);

	u64 max_key = n + 1;
	max_deg_plus_1 = max_deg + 1;
	u64 maxhmsize = compute_max_hmap_size(max_deg);
	u64 HASH_tbl_size = maxhmsize < n ? maxhmsize + 1 : max_key;
	u64 logsize = compute_logsize(max_key);

	for (int i = 0; i < p; i++) {
		posix_memalign((void**)&thread_HASHmap[i], ALN,
			(HASH_tbl_size) * sizeof(neigh_comm_map_t));
		assert(thread_HASHmap[i] != NULL);
		posix_memalign((void**)&thread_HASH_nonempty_loc[i], ALN,
			(max_deg_plus_1) * sizeof(u64));
		assert(thread_HASH_nonempty_loc[i] != NULL);
	}
	one_over_m2 = 1.0 / total_graph_edge_weight;
	printf("maxdeg %llu m = %f\n", max_deg, total_graph_edge_weight);

	u16 level = 0;
    u64 start = getMilliCount();
    louvain_inner_loop(HASH_tbl_size, maxhmsize, n + 1, p, community, G,
        logsize, thread_HASHmap, thread_HASH_nonempty_loc,
        comm_lock, vertex_lock);
    printf("Final modularity %f\n", modularity);
    u64 end = getMilliSpan(start);
    cout << "level " << level << ", time: " << (double)end << "ms "
        << (double)end / 1000.0 << "s\n";


	//allocate distance array
	deallocate_memory();
	_mm_free(comm_lock);
	_mm_free(vertex_lock);
	return 0;
}
