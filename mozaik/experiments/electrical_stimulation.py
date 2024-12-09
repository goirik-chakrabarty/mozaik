import os
import numpy as np
from mozaik.experiments import Experiment
from parameters import ParameterSet
from mozaik.stimuli import InternalStimulus
from collections import OrderedDict
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet
import random
from mozaik import load_component

class RandomSingleNeuronStepCurrentInjection(Experiment):
    """
    This experiment schedules many single neuron injections, whereby neurons are randomly picked from the set defined by 
    population selector provided in stimulation_configuration.
    
    Parameters
    ----------
    duration : float (ms)
          The duration of the current injection

    current : float (mA)
          The magnitude on injected current
          
    sheet : str
          The sheet from which the neurons are selected
    
    num_neurons : int
          The number of neurons that will be stimulated

    num_trials : int
          Number of trials to repeat each repetition (the order of stimulations will be randomised)
    
    stimulation_configuration : ParameterSet
          The population selector from which the neurons are randomly selected
    """


    required_parameters = ParameterSet({
            'duration': float,
            'current' : float,
            'sheet' : str,
            'num_neurons' : int,
            'num_trials' : int, 
            'stimulation_configuration' : ParameterSet,
        })

    
    def __init__(self,model,parameters):
            Experiment.__init__(self, model,parameters)
            from mozaik.sheets.direct_stimulator import Depolarization

            population_selector = load_component(self.parameters.stimulation_configuration.component)
            ids = population_selector(model.sheets[sheet],self.parameters.population_selector.params).generate_idd_list_of_neurons()

            ids = random.sample(ids,self.parameters.num_neurons)

            d  = OrderedDict()

            for i in range(0,self.parameters.num_trials):
                random.shuffle(ids)
                for idd in ids:

                    p = MozaikExtendedParameterSet({
                                                    'current' : self.parameters.current,
                                                    'population_selector' : MozaikExtendedParameterSet({
                                                         'component' : 'IDList',
                                                         'list_of_ids' : [idd]
                                                    })
                                                  })

                    d[self.parameters.sheet] = [Depolarization(model.sheets[self.parameters.sheet],p)]
                    
                    self.direct_stimulation = [d]
                    self.stimuli.append(
                                InternalStimulus(   
                                                    frame_duration=self.parameters.duration, 
                                                    duration=self.parameters.duration,
                                                    trial=0,
                                                    direct_stimulation_name='Injection',
                                                    direct_stimulation_parameters = p
                                                )
                                        )
