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


#include <iostream>
#include <algorithm>
#include <string.h>
#include <fstream>
#include <time.h>
#include <sys/types.h>
#include <sys/timeb.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <emmintrin.h>
#include <vector>
#include <map>

using namespace std;

#define u8 unsigned char
#define u16 unsigned short
#define u64 unsigned long long int
#define u32 unsigned int

#define MIN(a,b) (a<b)?a:b
#define MAX(a,b) (a>b)?a:b


/******************************* Timing Function ***************************/
unsigned long long getMilliCount() {
	timeb tb;
	ftime(&tb);
	unsigned long long nCount = tb.millitm + (tb.time & 0xfffffffff) * 1000;
	return nCount;
}
unsigned long long getMilliSpan(unsigned long long nTimeStart) {
	long long int nSpan = getMilliCount() - nTimeStart;
	if (nSpan < 0)
		nSpan += 0x100000 * 1000;
	return nSpan;
}

/********************************** Parameter processing ***************************/
char* getParamVal(int argc, char** argv, char* param) {
	for (int i = 1; i < argc - 1; i++)
		if (!strcasecmp(argv[i], param))
			return argv[i + 1];
	return NULL;
}
//Align edges to 64 bit boundary for making them cache efficient
#ifndef ALN
#define ALN 64
#endif

/*  CHUNK_SIZE to control load-balancing, emperically find the best for your graph.
	Changing this variable will change load-balancing in the code
	granularity of work.
*/
#ifndef CHUNK_SIZE
#define CHUNK_SIZE 4
#endif

// Turn WEIGHTED on to make the graph weighted (i.e., hinting the each edge has a weight associated with it)
//#define WEIGHTED


// Defines switching point between pull and push
#ifndef SWITCH_POINT
#define SWITCH_POINT 5
#endif

#define SORT_EDGES           //Sort edges turned on by default
#define DEDUPLICATE          //Turn this on to debuplicate the edges
/* Turn this on to make a directed graph undirected*. For Hybrid algorithm,
   this FLAG needs to be on always.
*/
#define REVERSE_EDGE        

//#define SORT_BY_DEGREE
#ifdef REVERSE_EDGE
#undef REMOVE_ZERO_DEGREE_NEIGBORS
#endif
	
//#define ANALYZE                     // Turn on to enable analyze             

double min_modularity = 0.000001;     // Modularity threshold
int num_chunk = 0;                    // Total number of chunks
double total_graph_edge_weight = 0.0; // Total edge weight
double one_over_m2 = 0.0;             // 1/total_edge_weight
double modularity = 0.0;              // Current modularity

u64 max_deg = 0;                      // Maximum degree of the graph
u64 max_deg_plus_1;                   // Max_deg + 1
u32 p;                                // Number of threads
u64 n, m;                             // Number of vertices and edges


// HASH related variable
static u64 max_key;

#ifdef ANALYZE
u64 pull_time = 0, push_time = 0;
u64 vertex_explored = 0;
u64 edges_explored = 0;
#endif

// Structure to represent edges
typedef struct edge_t {
	u64 u, v;
	double edge_weight;

} edge_t;

// Structure to represent neighbors
typedef struct neighbor_t {
	u64 neigh_id;
	double edge_weight;
	bool operator<(const neighbor_t& val) const {
		return neigh_id < val.neigh_id;
	}
} neighbor_t;


// Structure of each vertex in the graph
struct graph_t {
	u64 deg;                    //degree
	double total_edge_weight;
	double self_loop_weight;
	neighbor_t* offset;         //adjacency list starting point in the adjMat array
};

// Structure for community info
typedef struct {
	double in;
	double tot;
} community_t;

// Structure for neighbor community map
typedef struct neigh_comm_map_t {
	u64 comm_id;
	double weight_sum;

} neigh_comm_map_t;

typedef struct HASH_loc_ptr {
	u32 HASH_table_size;
	u64* non_empty_loc;
} HASH_loc_ptr_t;


neighbor_t* adjMat;                // Adjacency list
graph_t* G;                        // Graph
community_t* community;            // Community list
u64* n2c;                          // Community id for each vertex

neigh_comm_map_t** vertex_HASHmap_ptr;  // Shared hash map
HASH_loc_ptr_t* vertex_HASH_loc_ptr;    // Shared non-empty hash location

//#define PRUNE                   // Turn on to enable vertex pruning
#ifdef PRUNE
u8* recompute;                    // Flag to set whether a vertex should get recomputed, or pruned
#endif

//#define PRUNE_PULL              // 
/* Comparator to SORT by degree   */
struct sortByDeg {
	bool operator()(const neighbor_t& a, const neighbor_t& b) {
		return G[a.neigh_id].deg > G[b.neigh_id].deg;
	}
} sortByDeg;