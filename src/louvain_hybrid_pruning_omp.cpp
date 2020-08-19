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


#include "louvain_hybrid_pruning_omp.h"

/********************************** HASHing function *******************************/

static inline u64 HASH(u64 x, const u64 HASHlen) {
	return (x * 0x9e3779b97f4a7c15) % (HASHlen);
	//return (x) % (HASHlen);
}
void HASH_update(neigh_comm_map_t* hmap, const double edge_weight,
	const u64 node_comm, const u64 best_comm, const u64 lsize,
	/*const u64 max_key,
	u64* neigh_pos, u32* num_comm*/
	HASH_loc_ptr_t* HASH_loc_ptr) {
	u64* neigh_pos = HASH_loc_ptr->non_empty_loc;
	u32* num_comm = &(HASH_loc_ptr->HASH_table_size);
	u64 l = HASH(node_comm, lsize);
	u64 nc = 0;
	for (; // init
		hmap[l].comm_id != node_comm && nc < lsize; // test
		l = ((l + 1) % lsize), nc++) {
	} // increment
	  //decrease weight for node_comm
	if (hmap[l].comm_id == node_comm) {
		hmap[l].weight_sum -= edge_weight;
		// if weight has become zero,
		if (hmap[l].weight_sum <= 0.) {
			hmap[l].comm_id = max_key;
			u64 loc = 0;
			while (neigh_pos[loc] != l)
				loc++;
			*num_comm = *num_comm - 1;
			neigh_pos[loc] = neigh_pos[*num_comm];
		}
	}

	nc = 0;
	//increase weight for best_comm
	u64 l2 = HASH(best_comm, lsize);
	for (; // init
		hmap[l2].comm_id != best_comm && hmap[l2].comm_id != max_key
		&& nc < lsize; // test
		l2 = ((l2 + 1) % lsize)) {
	} // increment
	if (hmap[l2].comm_id == best_comm) {
		hmap[l2].weight_sum += edge_weight;
	}
	else if (hmap[l2].comm_id == max_key) {
		hmap[l2].comm_id = best_comm;
		hmap[l2].weight_sum = edge_weight;
		// need to lock the neibhgor pos, pass the struct as a whole for that
		neigh_pos[*num_comm] = l2;
		*num_comm = *num_comm + 1;
	}
}
static inline u32 HASH_insert(const u64 ck, neigh_comm_map_t* hmap,
	const double edge_weight, const u64 lsize, 
	u64* neigh_pos, u32* num_comm) {
	u64 nc = 0;
	u64 l = HASH(ck, lsize);
	for (; // init
		hmap[l].comm_id != ck && hmap[l].comm_id != max_key && nc < lsize; // test
		l = ((l + 1) % lsize), nc++) {
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
		printf("HASH_insert error: ck=%llu, l=%llu hmap[l].comm_id=%llu\n", ck,
			l, hmap[l].comm_id);
	return nc;
}

/*
 Function removes duplicate elements from a sorted array and
 returns new size of modified array.
*/
u64 removeDuplicates(neighbor_t* adj, u64 length) {
	if (length == 0 || length == 1)
		return length;
	neighbor_t* temp = (neighbor_t*)malloc(sizeof(neighbor_t) * length);
	
	u64 j = 0;
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
	for (u64 i = 0; i < j; i++)
		adj[i] = temp[i];
	free(temp);
	return j;
}

/* Initialize community membership and hash tables */
void louvain_init_community(graph_t* __restrict g, const u64 n,
	community_t* __restrict community, const int num_thread) {
	double my_weight[num_thread];
	u64 mdeg[num_thread];
	
	#pragma parallel
	// this can be done in parallel
	for (int t = 0; t < num_thread; t++) {
		my_weight[t] = 0.;
		mdeg[t] = 0;
	}
	#pragma omp for schedule(static,CHUNK_SIZE)
	for (u64 i = 0; i < n; i++)
//#pragma omp parallel for
	//for (int t = 0; t < num_chunk; t++) 
	{
		int thread_id = omp_get_thread_num();
		//u64 start = t * CHUNK_SIZE;
		//u64 end = (start + CHUNK_SIZE <= n) ? (start + CHUNK_SIZE) : n;
		//for (int i = start; i < end; i++) 
		{
			n2c[i] = i;
			u64 deg = g[i].deg;
			if (deg == 0) {
				community[i].tot = 0.;
				community[i].in = 0.;
				continue;
			}
			n2c[i] = i;
			neighbor_t* edge_list = g[i].offset;
			double edge_weight = 0.0; //w(u)
			double self_weight = 0.0;
			for (int j = 0; j < deg; j++) {
				edge_weight += edge_list[j].edge_weight;
				if (edge_list[j].neigh_id == i)
					self_weight = edge_list[j].edge_weight;
			}
			g[i].total_edge_weight = edge_weight;
			g[i].self_loop_weight = self_weight;
			community[i].tot = edge_weight;
			community[i].in = self_weight;
			my_weight[thread_id] += edge_weight;
			if (mdeg[thread_id] < deg)
				mdeg[thread_id] = deg;
		}
	}
	// this can be done with parallel reduction
	for (int t = 0; t < num_thread; t++) {
		total_graph_edge_weight += my_weight[t];
		max_deg = MAX(mdeg[t], max_deg);
	}
}
/* Phase 1 (inner loop) of Louvain */
void louvain_inner_loop_hybrid(const u64 max_key,
	const int num_thread, community_t* restrict community, graph_t const* const restrict g,
	omp_lock_t* restrict HASH_entry_lock)
{
	//compute modularity
	/*********************************************************************/
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
			//#pragma omp atomic
			tot_sum_c += (tm2 * tm2);
			//#pragma omp atomic
			in_sum_c += community[i].in;
		}
	}
	modularity = (in_sum_c - tot_sum_c * one_over_m2) * one_over_m2;
	printf("Initial modularity %f\n", modularity);
	/*********************************************************************/
	double cur_mod;
	int iter = 0;
#ifdef ANALYZE
	u64 start_time = 0;
#endif
	do {
		cur_mod = modularity;
		//if iteration count < SWITCHING POINT use pull based
		/*****************************************************************/

		if (iter < SWITCH_POINT) 
		{
#ifdef ANALYZE
			start_time = getMilliCount();
#endif
			//#pragma omp for schedule (static,1)
#pragma omp parallel for schedule(static,1)
			for (int t = 0; t < num_chunk; t++) {
				u64 start = t * CHUNK_SIZE;
				u64 end = (start + CHUNK_SIZE <= N) ? (start + CHUNK_SIZE) : N;
				for (u64 i = start; i < end; i++) {

					const u64 deg = g[i].deg;
					if (deg == 0) continue;
#ifdef PRUNE_PULL
					if (recompute[i] == 0) continue;
#endif

#ifdef ANALYZE
					vertex_explored++;
#endif
					neigh_comm_map_t* node_HASHmap = vertex_HASHmap_ptr[i];
					u64* neigh_pos = vertex_HASH_loc_ptr[i].non_empty_loc;
					u32* num_comm = &(vertex_HASH_loc_ptr[i].HASH_table_size);

					u64 node_comm = n2c[i];

					// compute NCW
					const u64 HASH_v = deg + 1;               // size of vertex i's hash table
					const u64 l = HASH(node_comm, HASH_v);          // hash location
					node_HASHmap[l].comm_id = node_comm;      // update hash location with node comm
					node_HASHmap[l].weight_sum = 0.;          // update hash location with edge weights
					                                          // save the non-empty location
					neigh_pos[0] = l;                         // neigh_pos[0] contains vertex's own community
					*num_comm = 1;

					// Traverse neighbors
					neighbor_t* edge_list = g[i].offset;
					for (u64 e = 0; e < deg; e++)
					{
						u64 neigh = edge_list[e].neigh_id;

						if (neigh != i)
						{
							_mm_prefetch((char*)(edge_list + e + 64), _MM_HINT_T0);

							u64 neigh_comm = n2c[neigh];

							HASH_insert(neigh_comm, node_HASHmap,
								edge_list[e].edge_weight, HASH_v, 
								neigh_pos, num_comm);
						}
					}
					// find the best community to move into
					u64 best_comm = node_comm;
					double best_increase = 0.;
					double best_eweight = 0.;
                    
                    
					double n_in_weight = g[i].total_edge_weight;   // outgoing edge weights
					double w_n_over_m2 = n_in_weight * one_over_m2;
					double neigh_weight_node_comm = node_HASHmap[l].weight_sum;
					const u32 len = *num_comm;

					// Iterating through the community
					for (u32 hl = 0; hl < len; hl++) {
						u64 HASH_loc = neigh_pos[hl];
						u64 comm_id = node_HASHmap[HASH_loc].comm_id; //read
						node_HASHmap[HASH_loc].comm_id = max_key;     //reset

						double wn2c = node_HASHmap[HASH_loc].weight_sum;

						double totc = community[comm_id].tot;

						double increase = (wn2c - totc * w_n_over_m2);
						if (increase > best_increase) 
						{
							best_comm = comm_id;
							best_eweight = wn2c;
							best_increase = increase;
						}
						
					}
					if (best_comm != node_comm) 
					{
						double loops_tmp = g[i].self_loop_weight;

#pragma omp atomic
						community[node_comm].tot -= n_in_weight;
#pragma omp atomic
						community[node_comm].in -= (2.0 * neigh_weight_node_comm + loops_tmp);

#pragma omp atomic
						community[best_comm].tot += n_in_weight;
#pragma omp atomic
						community[best_comm].in += (2.0 * best_eweight + loops_tmp);

						n2c[i] = best_comm;

#ifdef PRUNE_PULL
		
						// any neighbor which is not in the new community of i will be recomputed 
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
#ifdef PRUNE_PULL
					else {
						recompute[i] = 0;
					}

#endif

				} //end vertex loop
			}
#ifdef ANALYZE
			pull_time += getMilliSpan(start_time);
#endif
		}

		//if iteration count == SWITCHING POINT initialize HASH table
		/*****************************************************************/
		else
		{
#ifdef ANALYZE
			start_time = getMilliCount();
#endif
			if (iter == SWITCH_POINT) 
			{

				u64 start_time = getMilliCount();
#pragma omp parallel for
//#pragma omp for schedule(static,1)
				for (int t = 0; t < num_chunk; t++) {
					u64 start = t * CHUNK_SIZE;
					u64 end = (start + CHUNK_SIZE <= N) ? (start + CHUNK_SIZE) : N;
					for (u64 i = start; i < end; i++) {
						const u64 deg = g[i].deg;
						if (deg == 0) continue;
#ifdef PRUNE_PULL
						recompute[i] = 1;
#endif
						neigh_comm_map_t* node_HASHmap = vertex_HASHmap_ptr[i];
						u64* neigh_pos = vertex_HASH_loc_ptr[i].non_empty_loc;
						u32* num_comm = &(vertex_HASH_loc_ptr[i].HASH_table_size);

						u64 node_comm = n2c[i];

						u64 HASH_v = deg + 1;
						u64 l = HASH(node_comm, HASH_v);
						node_HASHmap[l].comm_id = node_comm;
						node_HASHmap[l].weight_sum = 0.;
						neigh_pos[0] = l;
						*num_comm = 1;
						neighbor_t* edge_list = g[i].offset;
						for (u64 e = 0; e < deg; e++)
						{
							u64 neigh = edge_list[e].neigh_id;

							if (neigh != i)
							{
								_mm_prefetch((char*)(edge_list + e + 64), _MM_HINT_T0);

								u64 neigh_comm = n2c[neigh];

								HASH_insert(neigh_comm, node_HASHmap,
									edge_list[e].edge_weight, HASH_v, 
									neigh_pos, num_comm);

							}
						}
					}
				}
#if 0
				printf("overhead %lld ms\n", getMilliSpan(start_time));
#endif
			}
			//if iteration count >= SWITCHING POINT use push based
			/*****************************************************************/
#pragma omp parallel for schedule(static, 1)
			for (int t = 0; t < num_chunk; t++) {

				u64 start = t * CHUNK_SIZE;
				u64 end = (start + CHUNK_SIZE <= N) ? (start + CHUNK_SIZE) : N;

				for (u64 i = start; i < end; i++) {
#ifdef PRUNE
					if (recompute[i] == 0) continue;
#endif
					const u64 deg = g[i].deg;
					if (deg == 0) continue;
#ifdef ANALYZE
					vertex_explored++;
#endif
					neigh_comm_map_t* node_HASHmap = vertex_HASHmap_ptr[i];
					u64* neigh_pos = vertex_HASH_loc_ptr[i].non_empty_loc;
					u32 num_comm = (vertex_HASH_loc_ptr[i].HASH_table_size);

					// find the best community to move into

					u64 node_comm = n2c[i];
					u64 best_comm = node_comm;
					double best_increase = 0.;
					double best_eweight = 0.;

					u32 best_comm_pos = neigh_pos[0];
					double n_in_weight = g[i].total_edge_weight;
					double w_n_over_m2 = n_in_weight * one_over_m2;
					double neigh_weight_node_comm = 0.;
					u32 node_pos = 0;
					int found = 0;
					for (u32 hl = 0; hl < num_comm; hl++) {
						u64 HASH_loc = neigh_pos[hl];
						u64 comm_id = node_HASHmap[HASH_loc].comm_id;
						double wn2c = node_HASHmap[HASH_loc].weight_sum;

						if (node_comm == comm_id)
						{
							node_pos = hl;
							neigh_weight_node_comm = wn2c;
							found = 1;
						}

						double totc = community[comm_id].tot;

						double increase = (wn2c - totc * w_n_over_m2);
#if 0
						cout << "vertex " << i << " may move from " << node_comm << " to " << comm_id << " increase " << increase << endl;
#endif
						if (increase > best_increase) {
							best_comm = comm_id;
							best_eweight = wn2c;
							best_increase = increase;
							best_comm_pos = hl;
						}
					}
					if (best_comm != node_comm) {
						double loops_tmp = g[i].self_loop_weight;

#pragma omp atomic
						community[node_comm].tot -= n_in_weight;
#pragma omp atomic
						community[node_comm].in -= (2.0 * neigh_weight_node_comm + loops_tmp);

#pragma omp atomic
						community[best_comm].tot += n_in_weight;
#pragma omp atomic
						community[best_comm].in += (2.0 * best_eweight + loops_tmp);

						n2c[i] = best_comm;

						if (found) {

							omp_set_lock(HASH_entry_lock + i);
							node_HASHmap[neigh_pos[node_pos]].comm_id = max_key;
							//update neigh_pos
							vertex_HASH_loc_ptr[i].HASH_table_size = vertex_HASH_loc_ptr[i].HASH_table_size - 1;
							neigh_pos[node_pos] = neigh_pos[num_comm - 1];
							omp_unset_lock(HASH_entry_lock + i);

						}
						// now go though all neighbors and change their HASH_map
						// and neighbor_pos_thing

						neighbor_t* edge_list = g[i].offset;
#ifdef PRUNE
						recompute[i] = 1;
#endif
						for (u64 e = 0; e < deg; e++)
						{
							// pointer to neighbor HASH map
							u64 neigh = edge_list[e].neigh_id;
							if (neigh != i) 
                            {

#ifdef ANALYZE
								edges_explored++;
#endif
								u64 neigh_deg = g[neigh].deg + 1;
								neigh_comm_map_t* neigbor_HASHmap = vertex_HASHmap_ptr[neigh];
#ifdef PRUNE
								if (n2c[neigh] != best_comm)
									recompute[neigh] = 1;
#endif
								//update_neighbor HASH
								//increase for best_comm, might cause a new community a
								//decrease for node_comm
								omp_set_lock(HASH_entry_lock + neigh);
								HASH_update(neigbor_HASHmap, edge_list[e].edge_weight, node_comm, best_comm, neigh_deg, &vertex_HASH_loc_ptr[neigh]);
								omp_unset_lock(HASH_entry_lock + neigh);
							}	//end of if(neigh!=i)
						}	// end of for (u64 e = 0; e < deg; e++)
					}	//if (best_comm != node_comm)
#ifdef PRUNE
					else {

						recompute[i] = 0;
					}
#endif
				}
			} //end of thread chunk loop
#ifdef ANALYZE
			push_time += getMilliSpan(start_time);
#endif

		}

		double in_sum_c = 0;
		double tot_sum_c = 0;
#pragma omp parallel for reduction(+:tot_sum_c) reduction(+:in_sum_c)
		for (u64 i = 0; i < n; i++)
		{
			if (community[i].tot > 0.0)
			{

				double tm2 = (community[i].tot);
				tot_sum_c += (tm2 * tm2);
				in_sum_c += community[i].in;
			}
		}
		modularity = (in_sum_c - tot_sum_c * one_over_m2) * one_over_m2;
		iter++;
        printf("iteration %d new modularity %f\n", iter, modularity);
	} while (modularity - cur_mod > min_modularity /*&& iter < 30*/);
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
	omp_set_dynamic(0);	// Explicitly disable dynamic teams

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

	printf("num_threads %d, num_chunk %d\n", p, num_chunk);
	//allocate memory for the graph
	G = (graph_t*)_mm_malloc((n) * sizeof(graph_t), ALN);
	assert(G != NULL);

	u64* ofs;
	assert(ofs = (u64*)malloc(sizeof(u64) * (n)));
	edge_t* edgeList;
	assert(edgeList = (edge_t*)malloc(sizeof(edge_t) * m));

	//initialize all degrees to zero
#pragma omp parallel for schedule(static, CHUNK_SIZE)
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

#ifdef REVERSE_EDGE
		G[v].deg++;
#endif //REVERSE_EDGE
		i++;
	}
	infile.close();
	expected_edges = i;

#ifdef REVERSE_EDGE
	m = i * 2;
#else
	m = i;
#endif

	u64 exs = ALN / sizeof(neighbor_t);
	//Pad adjMat to be a multiple of ALN factor
	adjMat = (neighbor_t*)_mm_malloc((m + exs * n + 1) * sizeof(neighbor_t),
		ALN);
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
	for (u64 i = 0; i < expected_edges; i++) {
		u64 j = edgeList[i].u;
		u64 k = ofs[j];
		adjMat[k].neigh_id = edgeList[i].v;
		adjMat[k].edge_weight = (double)edgeList[i].edge_weight;
		ofs[j] = k + 1;
#ifdef REVERSE_EDGE
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
	free(edgeList);
	free(ofs);
	n2c = (u64*)_mm_malloc((n) * sizeof(u64), ALN);
	assert(n2c != NULL);

	// community info
	community = (community_t*)_mm_malloc((n) * sizeof(community_t), ALN);
	assert(community != NULL);



	omp_lock_t* restrict
		HASH_entry_lock = (omp_lock_t*)_mm_malloc(sizeof(omp_lock_t) * n, ALN);
	assert(HASH_entry_lock != NULL);

#pragma omp parallel for
	for (u64 i = 0; i < n; i++) {
		omp_init_lock(HASH_entry_lock + i);
	}
	vertex_HASHmap_ptr = (neigh_comm_map_t**)malloc((n) * sizeof(neigh_comm_map_t*));
	assert(vertex_HASHmap_ptr != NULL);
	vertex_HASH_loc_ptr = (HASH_loc_ptr_t*)malloc((n) * sizeof(HASH_loc_ptr_t));
	assert(vertex_HASH_loc_ptr != NULL);
	
    max_key = n + 1;

#ifdef PRUNE
	posix_memalign((void**)&recompute, ALN, (n) * sizeof(u8));
#endif

    #pragma omp parallel for schedule (static, CHUNK_SIZE)
	for (u64 i = 0; i < n; i++) {
		u64 len = G[i].deg + 1;
		posix_memalign((void**)&vertex_HASHmap_ptr[i], ALN, (len) * sizeof(neigh_comm_map_t));
		assert(vertex_HASHmap_ptr[i] != NULL);
		neigh_comm_map_t* HASH_map_ptr = vertex_HASHmap_ptr[i];
		for (u64 j = 0; j < len; j++) {
			HASH_map_ptr[j].comm_id = max_key;
		}
		posix_memalign((void**)&(vertex_HASH_loc_ptr[i].non_empty_loc), ALN, (len) * sizeof(u64));
		assert(vertex_HASH_loc_ptr[i].non_empty_loc != NULL);
#ifdef PRUNE
		recompute[i] = 1;
#endif
	}
	louvain_init_community(G, n, community, p);

	one_over_m2 = 1.0 / total_graph_edge_weight;
	printf("maxdeg %llu m = %f\n", max_deg, total_graph_edge_weight);

	unsigned long long start = 0, end = 0;
	start = getMilliCount();
	louvain_inner_loop_hybrid(max_key, p, community, G, HASH_entry_lock);
	printf("Final modularity %f\n", modularity);
	end = end + getMilliSpan(start);
    
	cout << "time: " << (double)end << "ms " << (double)end / 1000.0 << "s\n";
#ifdef ANALYZE
	cout << "pull time " << (double)pull_time << "ms push time " << push_time << " ms" << endl;
	cout << "vertices explored " << vertex_explored << " edges explored " << edges_explored << endl;
#endif
	//free memory
	_mm_free(HASH_entry_lock);

	_mm_free(n2c);
	_mm_free(community);
	_mm_free(adjMat);

	for (u64 i = 0; i < n; i++) {
		free(vertex_HASHmap_ptr[i]);
		free(vertex_HASH_loc_ptr[i].non_empty_loc);
	}
	free(vertex_HASHmap_ptr);
	free(vertex_HASH_loc_ptr);
#ifdef PRUNE
	free(recompute);
#endif
	_mm_free(G);
	return 0;
}
