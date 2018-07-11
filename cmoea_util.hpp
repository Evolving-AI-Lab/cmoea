/*
 * cmoea_util.hpp
 *
 *  Created on: Jun 6, 2017
 *      Author: joost
 */

#ifndef MODULES_CMOEA_CMOEA_UTIL_HPP_
#define MODULES_CMOEA_CMOEA_UTIL_HPP_

#include <modules/misc/common_compare.hpp>

namespace sferes {
namespace cmoea {

void calculate_bin_fitness(const std::vector<float>& task_performance, std::vector<float>& bin_fitness){
    unsigned nr_of_bins = pow(2, task_performance.size()) - 1;
    bin_fitness.clear();
    std::vector<float> fitness_combination;
    for(unsigned i=0; i<nr_of_bins; ++i){
        fitness_combination.clear();
        for(unsigned j=0; j<task_performance.size(); ++j){
            if(int(i/(1<<j))%2 == 0){
                fitness_combination.push_back(task_performance[j]);
            }
        }
        float local_fitness = compare::average(fitness_combination);

        dbg::out(dbg::info, "binfit") << "obj: " << i << " fitness " << local_fitness << std::endl;
        bin_fitness.push_back(local_fitness);
    }
}

void calculate_bin_fitness_mult(const std::vector<float>& task_performance, std::vector<float>& bin_fitness){
    unsigned nr_of_bins = pow(2, task_performance.size()) - 1;
    bin_fitness.clear();
    std::vector<float> fitness_combination;
    for(unsigned i=0; i<nr_of_bins; ++i){
        fitness_combination.clear();
        for(unsigned j=0; j<task_performance.size(); ++j){
            if(int(i/(1<<j))%2 == 0){
                fitness_combination.push_back(task_performance[j]);
            }
        }
        float local_fitness = compare::mult(fitness_combination);

        dbg::out(dbg::info, "binfit") << "obj: " << i << " fitness " << local_fitness << std::endl;
        bin_fitness.push_back(local_fitness);
    }
}
    
// Not an efficient implementation, but it at least guarentees that the objectives returned here
// are the objectives associated with that bin
std::vector<int> getTaskIndices(int bin_index, int nr_of_tasks){
    std::vector<int> result;
    unsigned nr_of_bins = pow(2, nr_of_tasks) - 1;
    for(unsigned j=0; j<nr_of_tasks; ++j){
        if(int(bin_index/(1<<j))%2 == 0){
            result.push_back(j);
        }
    }
    return result;
}
    
std::string getTaskIndicesStr(int bin_index, int nr_of_tasks){
    std::vector<int> temp = getTaskIndices(bin_index, nr_of_tasks);
    std::string result = "";
    for(unsigned i=0; i<temp.size(); ++i){
        result += boost::lexical_cast<std::string>(temp[i]);
        if(i<temp.size()-1){
            result += "_";
        }
    }
    return result;
}
    
unsigned getNrOfObjectives(unsigned nr_of_bins){
    unsigned result = 0;
    nr_of_bins+=1;
    while(nr_of_bins != 1){
        nr_of_bins = nr_of_bins >> 1;
        result+=1;
    }
    return result;
}

} // namespace cmoea
} // namespace sferes

#endif /* MODULES_CMOEA_CMOEA_UTIL_HPP_ */
