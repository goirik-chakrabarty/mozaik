import os
import numpy as np
from mozaik.experiments import Experiment
from mozaik.experiments.vision import VisualExperiment
from parameters import ParameterSet
from mozaik.stimuli import InternalStimulus
from collections import OrderedDict
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet
from mozaik import load_component
import mozaik.stimuli.vision.topographica_based as topo


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

    experiment_random_seed : int
          Random seed of the experiment

    """

    required_parameters = ParameterSet(
        {
            "duration": float,
            "current": float,
            "sheet": str,
            "num_neurons": int,  # initialize afferent layer 4 projections
            "num_trials": int,
            "stimulation_configuration": ParameterSet,
            "experiment_random_seed": int,
        }
    )

    def __init__(self, model, parameters):
        Experiment.__init__(self, model, parameters)
        from mozaik.sheets.direct_stimulator import Depolarization

        rng = np.random.default_rng(self.parameters.experiment_random_seed)

        population_selector = load_component(
            self.parameters.stimulation_configuration.component
        )
        ids = population_selector(
            model.sheets[self.parameters.sheet],
            self.parameters.stimulation_configuration.params,
        ).generate_idd_list_of_neurons()

        ids = rng.choice(ids, size=self.parameters.num_neurons, replace=False)

        self.direct_stimulation = []

        for i in range(0, self.parameters.num_trials):
            rng.shuffle(ids)
            for idd in ids:

                p = MozaikExtendedParameterSet(
                    {
                        "current": self.parameters.current,
                        "population_selector": MozaikExtendedParameterSet(
                            {
                                "component": "mozaik.sheets.population_selector.IDList",
                                "params": MozaikExtendedParameterSet(
                                    {"list_of_ids": [idd]}
                                ),
                            }
                        ),
                    }
                )

                d = OrderedDict()

                d[self.parameters.sheet] = [
                    Depolarization(model.sheets[self.parameters.sheet], p)
                ]

                self.direct_stimulation.append(d)
                self.stimuli.append(
                    InternalStimulus(
                        frame_duration=self.parameters.duration,
                        duration=self.parameters.duration,
                        trial=i,
                        direct_stimulation_name="Injection",
                        direct_stimulation_parameters=p,
                    )
                )


class RandomSingleNeuronStepCurrentInjectionDuringDriftingSinusoidalGratingStimulation(
    VisualExperiment
):
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
          Number of trials to repeat each combinations of orientation, contrast and stimulated neuron (the order of stimulated neurons will be randomised)

    grating_spatial_frequency : float
                      Spatial frequency of the grating.

    grating_temporal_frequency : float
                      Temporal frequency of the grating.

    grating_contrasts : list
              List of contrasts (expressed as % : 0-100%) of the grating.

    grating_num_orientations : int
              Number of evenly spaced orientations to show.

    stimulation_configuration : ParameterSet
          The population selector from which the neurons are randomly selected

    experiment_random_seed : int
          Random seed of the experiment
    """

    required_parameters = ParameterSet(
        {
            "duration": float,
            "current": float,
            "sheet": str,
            "num_neurons": int,
            "num_trials": int,
            "grating_spatial_frequency": float,
            "grating_temporal_frequency": float,
            "grating_contrasts": list,
            "grating_num_orientations": int,
            "stimulation_configuration": ParameterSet,
            "experiment_random_seed": int,
        }
    )

    def generate_stimuli(self):

        from mozaik.sheets.direct_stimulator import Depolarization

        rng = np.random.default_rng(self.parameters.experiment_random_seed)

        population_selector = load_component(
            self.parameters.stimulation_configuration.component
        )
        ids = population_selector(
            self.model.sheets[self.parameters.sheet],
            self.parameters.stimulation_configuration.params,
        ).generate_idd_list_of_neurons()

        ids = rng.choice(ids, size=self.parameters.num_neurons, replace=False)

        self.direct_stimulation = []

        for c in self.parameters.grating_contrasts:
            for i in range(0, self.parameters.grating_num_orientations):
                for t in range(0, self.parameters.num_trials):
                    rng.shuffle(ids)
                    for idd in ids:

                        p = MozaikExtendedParameterSet(
                            {
                                "current": self.parameters.current,
                                "population_selector": MozaikExtendedParameterSet(
                                    {
                                        "component": "mozaik.sheets.population_selector.IDList",
                                        "params": MozaikExtendedParameterSet(
                                            {"list_of_ids": [idd]}
                                        ),
                                    }
                                ),
                            }
                        )

                        d = OrderedDict()

                        d[self.parameters.sheet] = [
                            Depolarization(self.model.sheets[self.parameters.sheet], p)
                        ]

                        self.direct_stimulation.append(d)
                        self.stimuli.append(
                            topo.FullfieldDriftingSinusoidalGrating(
                                frame_duration=self.frame_duration,
                                size_x=self.model.visual_field.size_x,
                                size_y=self.model.visual_field.size_y,
                                location_x=0.0,
                                location_y=0.0,
                                background_luminance=self.background_luminance,
                                contrast=c,
                                duration=self.parameters.duration,
                                density=self.density,
                                trial=t,
                                orientation=np.pi
                                / self.parameters.grating_num_orientations
                                * i,
                                spatial_frequency=self.parameters.grating_spatial_frequency,
                                temporal_frequency=self.parameters.grating_temporal_frequency,
                                direct_stimulation_name="Injection",
                                direct_stimulation_parameters=p,
                            )
                        )

    def do_analysis(self, data_store):
        pass
