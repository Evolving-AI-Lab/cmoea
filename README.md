## CMOEA

This repository contains the source code for the sferes2 CMOEA module, which is a multiobjective evolutionary algorithm designed to solve multimodal problems. The accompanying publication is:

[Huizinga J](http://www.cs.uwyo.edu/~jhuizing/), [Clune J](http://jeffclune.com). (2018). ["Evolving Multimodal Robot Behavior via Many Stepping Stones with the Combinatorial Multi-Objective Evolutionary Algorithm"](https://arxiv.org/abs/1807.03392). arXiv:1807.03392.

**If you use this software in an academic article, please consider citing:**

    @article{huizinga2018evolving, 
        title={Evolving Multimodal Robot Behavior via Many Stepping Stones with the Combinatorial Multi-Objective Evolutionary Algorithm}, 
        author={Huizinga, Joost and Clune, Jeff}, 
        journal={arXiv preprint arXiv:1807.03392}, 
        year={2018}
    }


## 1. Installation

This CMOEA module is intended to be used with the [sferes2](https://github.com/sferes2/sferes2) framework. In addition, it depends on two other sferes2 modules, the [datatools](https://github.com/JoostHuizinga/datatools) module and the [nsgaext](https://github.com/JoostHuizinga/nsgaext) module.

First, download the sferes2 framework (you can skip this step if you already have sferes2 installed):

    git clone https://github.com/sferes2/sferes2.git

Next, download the two required modules by navigating to the modules inside sferes2, and cloning the two dependecies.

    cd sferes2/modules
    git clone git@github.com:JoostHuizinga/nsgaext.git
    git clone git@github.com:JoostHuizinga/datatools.git

Laslty, download the CMOEA module.

    git clone git@github.com:Evolving-AI-Lab/cmoea.git


You are now ready to build experiments with CMOEA! To run the example experiment provided in this repository, first navigate to the sferes2 root folder.

    cd ..

Now copy the example experiment to the experiments folder:

    cp -r modules/cmoea/cmoea_example exp


And configure and build the experiment:

    ./waf configure
    ./waf --exp cmoea_example

If the build was successful, you can now run the experiment with:

    build/exp/cmoea_example/maze_rls_cmoea


## 2. Licenses
The code in this repository is licensed under the MIT License.


## 3. Questions?
If you have questions/suggestions, please feel free to [email](mailto:joost.hui@gmail.com) or create github issues. 
