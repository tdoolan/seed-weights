
### Assumptions

* Locality (as with HyperNEAT)
* Known = (Input V Output) + seed_dim radius
    * Only weights to known values are updated
    * Number of updates counted, ratio's determine relevance
        * Instead use hold-out set to find ratio's?
* Precomputed float output?

### Open

Does increasing the seed network size adversely affect the model results due
to a reduced number of questionmarks (and thus more 'garbage' updates)?
**When does this happen?**

Ratios currently only over output nodes. Model seperately for inputs? More
accurate informed updates, but partial with respect to output node.

Consistent boxes solution?  
Data selection dependent? Distribute input areas for train and val?

Multiple seed outputs: More complex models?  
Currently suited for single indicator output type tasks?

Convergence to different solution types? Only in theory?

### Closed

Fitness_func: dx, dy to target?  
Nope, because of seed growth, there is an inherent bias towards low dx, dy solutions

Just grow the seed before even modeling the weights?  
No notable consistent substrate improvement  
See results:1

Lower thres improves 3x3?  
Nope, requires more accurate modeling (than max)  
See results:2

