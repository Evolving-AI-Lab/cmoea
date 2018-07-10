/*
 * cmoea_util.hpp
 *
 *  Created on: Jun 6, 2017
 *      Author: joost
 */

#ifndef MODULES_CMOEA_CMOEA_UTIL_HPP_
#define MODULES_CMOEA_CMOEA_UTIL_HPP_
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

} // namespace cmoea
} // namespace sferes

#endif /* MODULES_CMOEA_CMOEA_UTIL_HPP_ */
