// This example relies on the following other modules:
// - misc
// - nsgaext
// - cmoea

// Include standard
#include <iostream>
#include <fstream>

// Include sferes
#include <sferes/phen/parameters.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/eval/eval.hpp>
#include <sferes/stat/best_fit.hpp>
#include <sferes/eval/parallel.hpp>
#include <sferes/modif/dummy.hpp>
#include <sferes/modif/diversity.hpp>
#include <sferes/run.hpp>
#include <sferes/ea/dom_sort_basic.hpp>

// Include modules
#include <modules/cmoea/cmoea_util.hpp>
#include <modules/cmoea/cmoea_nsga2.hpp>
#include <modules/nsgaext/dom_sort_no_duplicates.hpp>
#include <modules/datatools/common_compare.hpp>

// Include local
#include <stat_cmoea.hpp>

/* NAMESPACES */
using namespace sferes;
using namespace sferes::gen::evo_float;


////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
///////////* Parameters for the experiment *////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////

/* Parameters for Sferes */
struct Params {
    
    // Parameters for the vector we are evolving
    struct evo_float {
        SFERES_CONST float cross_rate = 0.1f;
        SFERES_CONST float mutation_rate = 0.1f;
        SFERES_CONST float eta_m = 15.0f;
        SFERES_CONST float eta_c = 10.0f;
        SFERES_CONST mutation_t mutation_type = polynomial;
        SFERES_CONST cross_over_t cross_over_type = sbx;
    };
    
    struct parameters {
        SFERES_CONST float min = 0.0f;
        SFERES_CONST float max = 1.0f;
    };
    
    struct cmoea_nsga {
        // Type of non-dominated comparison that will be used by CMOEA NSGA
        typedef sferes::ea::_dom_sort_basic::non_dominated_f non_dom_f;
    };

    struct cmoea {
        // Size of every CMOEA bin
        static const unsigned bin_size = 10;
        
        // Number of CMOEA bins
        static const unsigned nb_of_bins = 7; // = (2 ^ numberOfTasks - 1)
        
        // Where fitness objective will be stored before non-dominated sorting
        static const unsigned obj_index = 0;
    };
    
    /* Parameters for the population */
    struct pop {
        // Required by sferes, but ignored by CMOEA
        SFERES_CONST unsigned size = 0;
        
        // The number of individuals create to initialize the population
        SFERES_CONST unsigned init_size = 10;
        
        // The maximum number of individuals to try an create at the same time.
        SFERES_CONST unsigned max_batch_size = 1024;

        // The number of individuals created every generation
        SFERES_CONST unsigned select_size = 100;
        
        // The number of generations for which to run the algorithm
        SFERES_CONST unsigned nb_gen = 200;
        
//        // A multiplier on the number of individuals to create a generation 1.
//        SFERES_CONST int initial_aleat = 1;

        // Frequency at which to dump the archive
        static const int dump_period = 200;
        
//        // Fequency at which to write a checkpoint
//        static const int checkpoint_period = 10;
    };

    struct ea {
    	typedef sferes::ea::dom_sort_basic_f dom_sort_f;
    };
    
    struct stats {
        static const size_t period = 1;
    };
};


////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////* Fitness Function *////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////

/**
 * Object to calculate the fitness of an individual over a set of randomly
 * generated mazes.
 */
SFERES_FITNESS(ExampleCmoeaFit, sferes::fit::Fitness) {
public:
    ExampleCmoeaFit(){
        
    }

    // Calculates behavioral distance between individuals
    template<typename Indiv>
    float dist(Indiv& ind) const {
        return compare::l1diff(_behavior, ind.fit().getBehavior());
    }


    // Evaluates the performance of an individual on all tasks
    template<typename Indiv>
    void eval(Indiv& ind) {
    	for(size_t i=0; i<ind.size(); ++i){
            _behavior.push_back(ind.data(i));
    	}
        float hard = ind.data(0) * ind.data(1) * ind.data(2);
        float easy = ind.data(3);
        float task0 = 0;
        float task1 = 0;
        float task2 = 0;
        if(easy > hard){
            task0 = easy;
            task1 = (1-hard)/2.0;
        } else {
            task0 = easy;
            task1 = hard;
        }
         if(hard > 0.9){
             task2 = ind.data(4);
         }
        
        _cmoea_task_performance.push_back(task0);
        _cmoea_task_performance.push_back(task1);
        _cmoea_task_performance.push_back(task2);
        
        cmoea::calculate_bin_fitness_mult(_cmoea_task_performance, _cmoea_bin_fitness);
        this->_objs.push_back(0);
        this->_objs.push_back(0);
    }

    ////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////
    //////////////* Fitness attributes *////////////////////
    ////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){
        dbg::trace trace("fit", DBG_HERE);
        sferes::fit::Fitness<Params,
			typename stc::FindExact<ExampleCmoeaFit<Params, Exact>,
			Exact>::ret>::serialize(ar, version);
        ar & BOOST_SERIALIZATION_NVP(_behavior);
        ar & BOOST_SERIALIZATION_NVP(_cmoea_task_performance);
        ar & BOOST_SERIALIZATION_NVP(_cmoea_bin_fitness);
        ar & BOOST_SERIALIZATION_NVP(_cmoea_bin_diversity);
        ar & BOOST_SERIALIZATION_NVP(_divIndex);
    }

    // Getters and setters for CMOEA
    std::vector<float> getBehavior() const { return _behavior;}
    std::vector<float> &getBinFitnessVector(){return _cmoea_bin_fitness;}
    float getBinFitness(size_t index){ return _cmoea_bin_fitness[index];}
    std::vector<float> &getBinDiversityVector(){return _cmoea_bin_diversity;}
    float getBinDiversity(size_t index){ return _cmoea_bin_diversity[index];}
    float getCmoeaObj(size_t index){ return _cmoea_task_performance[index];}
    
    void initBinDiversity(){
        if(_cmoea_bin_diversity.size() != Params::cmoea::nb_of_bins){
            _cmoea_bin_diversity.resize(Params::cmoea::nb_of_bins);
        }
    }
    
    void setBinDiversity(size_t index, float div){
        dbg::check_bounds(dbg::error, 0, index, _cmoea_bin_diversity.size(),
        		DBG_HERE);
        _cmoea_bin_diversity[index] = div;
    }

    

protected:
    // Vector of robot (end) points over all mazes
    std::vector<float> _behavior;
    // Performance on each task, as required for cmoea
    std::vector<float> _cmoea_task_performance;
    // Performance on each combination of tasks, as required for cmoea
    std::vector<float> _cmoea_bin_fitness;
    // The diversity score of an individual, as calculated by cmoea
    std::vector<float> _cmoea_bin_diversity;
    // The index of the diversity objective in the objective array
    int _divIndex;
};

////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
//////////////////* Main Program *//////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////

/* Sets up and runs the experiment. */
int main(int argc, char **argv) {
    std::cout << "Entering main: " << std::endl;
    
    time_t t = time(0) + ::getpid();
    std::cout<<"seed: " << t << std::endl;
    srand(t);
    sferes::misc::seed(t);

    /* Fitness function to use (a class, defined above), which makes use of the
     * Params struct. */
    typedef ExampleCmoeaFit<Params> fit_t;

    // The genotype.
    typedef gen::EvoFloat<5, Params> gen_t;
    
    // The phenotype
    typedef phen::Parameters<gen_t, fit_t, Params> phen_t;

    //Behavioral distance based only on current population.
    typedef modif::Diversity<> mod_t;

    // What statistics should be gathered
    typedef boost::fusion::vector<stat::StatCmoea<phen_t, Params> > stat_t;

    // The evaluator for the network
    typedef eval::Eval<Params> eval_t;
    
    // CMOEA with NSGA-II non-dominated selection within each bin
    typedef ea::CmoeaNsga2<phen_t, eval_t, stat_t, mod_t, Params> ea_t;

    ea_t ea;
    
    run_ea(argc, argv, ea);

    /* Record completion (makes it easy to check if the job was preempted). */
    std::cout << "\n==================================" << \
            "\n====Evolutionary Run Complete!====" << \
            "\n==================================\n";
    return 0;
}
