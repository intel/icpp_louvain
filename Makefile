#
# Copyright (C) 2020 Intel Corporation
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# SPDX-License-Identifier: BSD-3-Clause
#

#provide number of OMP threads to spawn
THREADS_COUNT=1

#dataset selection, edge list format
DATASET=hollywood-2009.elist

#includes
INC=./include

#compilation & linking flags
#CXXFLAGS=-O3 -xHost -parallel -ansi-alias -unroll -finline -restrict -qopt-report -funroll-loops -fomit-frame-pointer -fPIC -I$(INC)
CXXFLAGS=-O3 -xHost -parallel -ansi-alias -unroll -finline -restrict -funroll-loops -fomit-frame-pointer -fPIC -I$(INC) -std=c++11
#LXXFLAGS=-qopenmp
LXXFLAGS=-fopenmp

#selected Intel compiler ICC2020
CC=icc

.PHONY: all
all: pre hybrid hybrid_prune pull pull_prune

pre: clean
	@ mkdir ./build

hybrid: ./src/louvain_hybrid_pruning_omp.cpp
	@ $(CC) -o ./build/lhybrid $^ $(CXXFLAGS) $(LXXFLAGS) 

hybrid_prune: ./src/louvain_hybrid_pruning_omp.cpp
	@ $(CC) -o ./build/lhybrid_prune $^ $(CXXFLAGS) $(LXXFLAGS) -DPRUNE -DPRUNE_PULL

pull: ./src/louvain_pull_prune_omp.cpp
	@ $(CC) -o ./build/lpull $^ $(CXXFLAGS) $(LXXFLAGS) -DREVERSE_EDGE

pull_prune: ./src/louvain_pull_prune_omp.cpp
	@ $(CC) -o ./build/lpull_prune $^ $(CXXFLAGS) $(LXXFLAGS) -DREVERSE_EDGE -DPRUNE

run-hybrid:
	@ OMP_NUM_THREADS=$(THREADS_COUNT) ./build/lhybrid -i ./dataset/$(DATASET) | tee ./$(DATASET).log

run-hybrid_prune:
	@ OMP_NUM_THREADS=$(THREADS_COUNT) ./build/lhybrid_prune -i ./dataset/$(DATASET) | tee ./$(DATASET).log

run-pull:
	@ OMP_NUM_THREADS=$(THREADS_COUNT) ./build/lpull -i ./dataset/$(DATASET) | tee ./$(DATASET).log

run-pull_prune:
	@ OMP_NUM_THREADS=$(THREADS_COUNT) ./build/lpull_prune -i ./dataset/$(DATASET) | tee ./$(DATASET).log


clean:
	@ rm -rf ./build
	@ if [ -f $(DATASET).log ]; then rm $(DATASET).log; fi

