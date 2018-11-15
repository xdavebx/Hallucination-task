# -*- coding: utf-8 -*-

from __future__ import division
from psychopy import core, visual, event, gui, monitors, data, contrib
import pandas as pd
import os
import yaml
import math
import numpy as np
import ctypes
import warnings
import matplotlib.pyplot as plt
from psychopy.iohub import launchHubServer
from psychopy.iohub.constants import EventConstants
from psychopy import prefs
import nidaqmx

def rgb_convert(rgb):
    return tuple(i * 2 - 1 for i in rgb)

class fakeAnalogOutputTask():

    def write(self, value, auto_start=False):

        print("Writing + {0}".format(value))

    def start(self):

        print("Starting DAQ")

    def stop(self):

        print("stopping DAQ")

class ConfidenceScale(object):

    def __init__(self, win, n_bars=5, bar_duration=1, ymin=-3, ymax=3, width=5, labels=True):

        # Vertices for bar stims
        self.vertex_list = self.calculate_vertices(n_bars, ymin, ymax, width)

        # Bar colours
        cmap = plt.cm.get_cmap('autumn')
        self.colours = cmap(np.arange(0, 1, 1. / float(n_bars)))
        self.colours = [rgb_convert(i[:3]) for i in self.colours]  # remove alpha

        # List of bar stimuli
        self.bar_stimuli = []

        # Labels - these get defined later
        self.bottom_label = None
        self.top_label = None

        # Confidence rating
        self.confidence_rating = 0

        # Create bars
        for i in range(n_bars):
            self.bar_stimuli.append(visual.ShapeStim(win, vertices=self.vertex_list[i], fillColor=(.0, .0, .0)))

            # Bottom bar label
            if labels and i == 0:

                pos = (self.vertex_list[i][0][0] - 3,
                       self.vertex_list[i][0][1] + (self.vertex_list[i][1][1] - self.vertex_list[i][0][1]) / 2.)
                self.bottom_label = visual.TextStim(win, text="Unsure", pos=pos)

            # Top bar label
            if labels and i == n_bars - 1:
                pos = (self.vertex_list[i][0][0] - 3,
                       self.vertex_list[i][0][1] + (self.vertex_list[i][1][1] - self.vertex_list[i][0][1]) / 2.)
                self.top_label = visual.TextStim(win, text="Certain", pos=pos)


    def calculate_vertices(self, n_bars, ymin, ymax, width):

        """
        Creates a list of vertices for rating scale bars

        Parameters
        ----------
        n_bars: Number of bars to create
        ymin: Y coordinate of the bottom edge of the lowest bar
        ymax: Y coordiante of the top edge of the highest bar
        width: Width of the bars

        Returns
        -------
        List of bar vertices

        """

        bar_height = ((ymax - ymin) - (ymax - ymin) / float(n_bars)) / n_bars
        gap = ((ymax - ymin) / float(n_bars)) / float(n_bars - 1)
        x_coords = (0 - width / 2., 0 + width / 2.)

        vertex_list = []

        for i in range(n_bars):

            bottom_y, top_y = (ymin + (bar_height + gap) * i, ymin + (gap * i) + (bar_height * (i + 1)))

            vertices = ((x_coords[0], bottom_y),
                        (x_coords[0], top_y),
                        (x_coords[1], top_y),
                        (x_coords[1], bottom_y))

            vertex_list.append(vertices)

        return vertex_list

    def fill_bars(self, n_bars):

        self.confidence_rating = n_bars

        for i in range(len(self.bar_stimuli)):

            if i < n_bars:
                self.bar_stimuli[i].fillColor = self.colours[i]

            else:
                self.bar_stimuli[i].fillColor = (.0, .0, .0)


    def draw(self):

        """
        Draws all the bars
        """

        for i in self.bar_stimuli:
            i.draw()

        if self.bottom_label is not None:
            self.bottom_label.draw()

        if self.top_label is not None:
            self.top_label.draw()


class PainConditoningTask(object):

    def __init__(self, config=None):

        # Load config
        # this sets self.config (the config attribute of our experiment class) to a dictionary containing all the values
        # in our config file. The keys of this dictionary correspond to the section headings in the config file, and
        # each value is another dictionary with keys that refer to the subheadings in the config file. This means you can
        # reference things in the dictionary by e.g. self.config['heading']['subheading']

        with open(config) as f:
            self.config = yaml.load(f)

        self.quest_settings = self.config['quest_settings']
        self.quest = None

        # Check quest numbers of trials
        if self.quest_settings['n_trials'] <= self.quest_settings['n_ignored']:
            raise ValueError("Number of QUEST trials must be greater than the number of ignored trials")

        # ------------------------------------#
        # Subject/task information and saving #
        # ------------------------------------#

        # Enter subject ID and other information
        dialogue = gui.Dlg()
        dialogue.addText("Subject info")
        dialogue.addField('Subject ID')
        dialogue.addText("Task settings")
        dialogue.addField('Run QUEST', initial=True)
        dialogue.show()


        # check that values are OK and assign them to variables
        if dialogue.OK:
            self.subject_id = dialogue.data[0]
            self.run_quest = dialogue.data[1]
        else:
            core.quit()

        # Recode blank subject ID to zero - useful for testing
        if self.subject_id == '':
            self.subject_id = '0'

        # Folder for saving data
        self.save_folder = self.config['directories']['saved_data']
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)

        self.save_path = '{0}/Subject{1}_data.csv'.format(self.save_folder, self.subject_id, data.getDateStr())

        # Data to be saved
        self.data = dict(trial_number=[],  # Trial number
                         stimulation_level=[],  # Detection probability
                         voltage=[],  # Actual voltage
                         response=[],  # Subject's response
                         confidence=[],  # Confidence rating
                         detected=[],  # Whether or not they detected the stimulus
                         session=[],  # Session number
                         block=[],  # Block number
                         catch=[])  # Whether this was a catch trial


        # -----------------------#
        # Monitor & window setup #
        # -----------------------#

        monitor = monitors.Monitor('monitor', width=40.92, distance=74)
        monitor.setSizePix((1024, 768))
        self.win = visual.Window(monitor=monitor, size=(1024, 768), fullscr=self.config['task_settings']['fullscreen'],
                                 allowGUI=False,
                                 color=rgb_convert((-1, -1, -1)),
                                 units='deg', colorSpace='rgb')
        self.win.mouseVisible = False  # make the mouse invisible
        self.frame_rate = 60

        # DAQ
        try:
            device = nidaqmx.libnidaqmx.Device(self.config['stimulation']['device'])
            device.reset()

            self.analogOutputTask = nidaqmx.AnalogOutputTask()
            self.analogOutputTask.create_voltage_channel('Dev1/ao0', min_val=-10.0, max_val=10.0)
        except:
            self.analogOutputTask = fakeAnalogOutputTask()


        # Keys used for making moves
        # self.response_keys = self.config['response keys']['response_keys']
        # self.response_phases = self.config['response keys']['response_phases']  # levels of the tree

        io = launchHubServer()
        self.keyboard = io.devices.keyboard

        self.pain_key = self.config['response_keys']['pain_key']
        self.no_pain_key = self.config['response_keys']['no_pain_key']

        # Clock
        self.clock = core.Clock()

        # --------#
        # Stimuli #
        # --------#

        # Fixation cross
        self.fixation = visual.TextStim(win=self.win, height=self.config['stimuli']['fixation_size'],
                                        color='white', text="+")

        # Question mark for catch trials
        self.question_mark = visual.TextStim(win=self.win, height=self.config['stimuli']['question_mark_size'],
                                             color='white', text="?")

        self.instruction_text = visual.TextStim(win=self.win, height=self.config['stimuli']['text_size'],
                                                color='white', wrapWidth=30)
        self.instruction_text.fontFiles = [self.config['fonts']['font_path']]  # Arial is horrible
        self.instruction_text.font = self.config['fonts']['font_name']

        self.confidence_scale = ConfidenceScale(self.win, ymin=-5, ymax=5, width=8)


        # ---------------------- #
        # Stimulation parameters #
        # ---------------------- #

        # Default levels - used if calibration isn't performed
        self.detection_levels = [self.quest_settings['default_p0'], self.quest_settings['default_p25'],
                                 self.quest_settings['default_p50'], self.quest_settings['default_p75']]

        # Voltage for practice
        self.practice_levels = [self.config['task_settings']['low'], self.config['task_settings']['high']]

        # Voltage that shouldn't be exceeded
        self.abs_max_voltage = self.config['stimulation']['absolute_maximum_voltage']


    def run(self):

        """
        Runs the experiment
        """

        # Number of trials at each level
        n_p0 = self.config['task_settings']['n_p0']
        n_p25 = self.config['task_settings']['n_p25']
        n_p50 = self.config['task_settings']['n_p50']
        n_p75 = self.config['task_settings']['n_p75']

        # Catch trial
        catch = self.config['task_settings']['catch_trials'] * self.config['task_settings']['n_sessions']
        

        # ---------------------------------------------------------------------------------------------------------#

        # Practice trials with yes/no responses
        self.main_instructions(self.load_instructions(self.config['instructions']['start_instructions']), continue_keys=self.pain_key)
        self.run_practice_block(n_stim=self.config['task_settings']['practice_n_stim'],
                                n_no_stim=self.config['task_settings']['practice_n_no_stim'], confidence=False)

        # Practice confidence ratings
        self.main_instructions(self.load_instructions(self.config['instructions']['confidence_rating_instructions']))
        confidence_practice = ConfidencePracticeTrial(self)

        # Run 3 trials
        for i in range(self.config['task_settings']['n_confidence_practice']):
            confidence_practice.run(confidence=True)

        # Practice trials with confidence ratings
        self.main_instructions(self.load_instructions(self.config['instructions']['confidence_practice_instructions']))
        self.run_practice_block(n_stim=self.config['task_settings']['practice_n_stim'],
                                n_no_stim=self.config['task_settings']['practice_n_no_stim'], confidence=True)

        if self.run_quest:
            # Run QUEST to get detection points

            quest_OK = False

            while not quest_OK:

                self.main_instructions(self.load_instructions(self.config['instructions']['quest_instructions']), continue_keys=self.pain_key)
                self.detection_levels = self.run_quest_procedure(self.quest_settings['n_trials'], self.quest_settings['start_value'],
                                                       self.quest_settings['start_sd'], self.quest_settings['max_value'],
                                                       self.quest_settings['beta'], self.quest_settings['delta'],
                                                       self.quest_settings['gamma'], self.quest_settings['grain'])
                quest_OK = self.main_instructions(["QUEST OK?\n"
                                       "Press YES key to continue, NO key to rerun QUEST\n\n"
                                       "25% detection probability level = {0}\n"
                                       "25% detection probability level = {1}\n"
                                       "50% detection probability level = {2}\n"
                                       "75% detection probability level = {3}\n".format(self.detection_levels[0],
                                                                                        self.detection_levels[1],
                                                                                        self.detection_levels[2],
                                                                                        self.detection_levels[3])],
                                       continue_keys=[self.pain_key], return_keys=[self.no_pain_key])

            # Run task
            self.main_instructions(self.load_instructions(self.config['instructions']['task_instructions']))

        block_number = 0  # count blocks

        # Loop through sessions
        for session in range(self.config['task_settings']['n_sessions']):

            print("Session {0} / {1}\n" \
                  "-----------------".format(session + 1, self.config['task_settings']['n_sessions']))

            # Loop through blocks
            for i in range(block_number, block_number + self.config['task_settings']['n_blocks']):
                self.run_block(n_p0[i], n_p25[i], n_p50[i], n_p75[i], catch=catch[i], session=session, block=i)
                block_number += 1

            # Break screen
            if session != self.config['task_settings']['n_sessions'] - 1:
                self.main_instructions(self.load_instructions(self.config['instructions']['break_instructions']))

            # End screen
            else:
                self.main_instructions(self.load_instructions(self.config['instructions']['end_instructions']))


    def run_block(self, n_p0, n_p25, n_p50, n_p75, catch=False, confidence=True, session=0, block=0):

        n_trials = n_p0 + n_p25 + n_p50 + n_p75

        levels = np.hstack([np.zeros(n_p0), np.ones(n_p25) * 1, np.ones(n_p50) * 2, np.ones(n_p75) * 3])
        levels = levels.astype(int)
        np.random.shuffle(levels)

        for i in range(n_trials):  # TRIAL LOOP - everything in here is repeated each trial

            print("Trial {0} / {1}\n" \
                  "Detection probability = {2}".format(i + 1, n_trials, levels[i] * 25))

            # Get stimulation level for this trial
            stim_level = self.detection_levels[levels[i]]

            # If it's a catch trial, reduce the voltage
            if catch:
                stim_level = self.config['stimulation']['catch_intensity']

            trial = ConditioningTrial(self, i, True, stim_level)
            trial.run(catch=catch, confidence=confidence)
            trial.save_data(self.save_path, session, block, levels[i], catch)


    def run_practice_block(self, n_stim, n_no_stim, confidence=False):

        n_trials = n_stim + n_no_stim

        levels = np.hstack([np.zeros(n_no_stim), np.ones(n_stim)])
        levels = levels.astype(int)
        np.random.shuffle(levels)

        for i in range(n_trials):  # TRIAL LOOP - everything in here is repeated each trial

            print("Trial {0} / {1}\n".format(i + 1, n_trials))

            # Get stimulation level for this trial

            print(levels[i])
            print(self.practice_levels)
            stim_level = self.practice_levels[levels[i]]

            trial = ConditioningTrial(self, i, True, stim_level)
            trial.run(catch=False, confidence=confidence)


    def run_quest_procedure(self, n_trials=10, start=20, start_sd=10, maxvalue=40, beta=3.5, delta=0.01, gamma=0.01, grain=0.01):

        print("RUNNING QUEST\n" \
              "-------------\n")

        startVal=start
        startValSd=start_sd
        pThreshold=0.75
        nTrials=n_trials
        minVal=0
        maxVal=maxvalue
        beta=beta
        delta=delta
        gamma=gamma
        grain=grain
        method='quantile'
        
        conditions = {
            'startVal': startVal, 'startValSd': startValSd,
            'minVal': 0, 'maxVal': maxVal, 'pThreshold': 0.75, 'gamma': gamma, 'delta': delta,
            'grain': grain, 'method': method, 'nTrials': n_trials,
        }
        self.conditions = [
            dict({'label': 'staircase_1'}.items() + conditions.items()),
            dict({'label': 'staircase_2'}.items() + conditions.items()),
        ]
        self.quest = data.MultiStairHandler(stairType='quest', method='random', conditions=self.conditions)
        t1 = self.quest.staircases[0].mean()
        t2 = self.quest.staircases[1].mean()
        sd1 = self.quest.staircases[0].sd()
        sd2 = self.quest.staircases[1].sd()
        beta1 = contrib.quest.QuestObject.beta(self.quest.staircases[0])
        beta1 = contrib.quest.QuestObject.beta(self.quest.staircases[1])
        
        t_mean = (t1 + t2) / 2
        beta_mean = (beta1 + beta2) / 2 

        fit = data.FitWeibull(t_mean, beta_mean, sems=1/n_trials)

        for n, stim_level in enumerate(self.quest):

            print("Calibration trial {0} / {1}\n" \
                  "Stimulation level = {2}\n" \
                  "-----------------------------".format(n + 1, n_trials, stim_level))

            response = None

            # Repeat if subject doesn't respond
            while response == None:
                trial = ConditioningTrial(self, n, True, max_voltage=stim_level)
                response = trial.run()

            # Add the response
            if response:
                print("Stimulation detected")
            else:
                print("Stimulation not detected")

            if n >= self.config['quest_settings']['n_ignored']:
                self.quest.addResponse(int(response))
                p0 = fit.inverse(0)
                p25 = fit.inverse(.25)
                p50 = fit.inverse(.50)
                p75 = fit.inverse(.75)
                print("1% detection probability level = {0}\n"
                      "25% detection probability level = {1}\n"
                      "50% detection probability level = {2}\n"
                      "75% detection probability level = {3}\n".format(self.quest.quantile(0.01), self.quest.quantile(.25), self.quest.quantile(.5), self.quest.quantile(.75)))


        return p0, p25, p50, p75


    def load_instructions(self, text_file):

        """
        Loads a text file containing instructions and splits it into a list of strings

        Args:
            text_file: A text file containing instructions, with each page preceded by an asterisk

        Returns: List of instruction strings

        """

        with open(text_file, 'r') as f:
            instructions = f.read()

        return instructions.split('*')

    def get_responses(self, response_event, response_keys, start_time=0, mapping=None):
        """
        Watches for keyboard responses and returns the response and RT if one is made

        Args:
            response_event: Iohub response event (must not be None)
            response_keys: Allowed keys
            start_time: Trial start time, allows calculation of RT from current time - start time
            mapping: Dictionary of keys to map responses to, one per response key. E.g response keys [a, l] could be mapped
                     to [1, 2] by providing the dictionary {'a': '1', 'l': '2'}

        Returns:
            response: key pressed (mapped if mapping provided)
            rt: response time (minus start time if provided)

        """
        for kb_event in response_event:
            if kb_event.key in response_keys:
                response_time = kb_event.time - start_time
                response = kb_event.key
                if mapping is not None:
                    response = mapping[response]
            else:
                response = None
                response_time = None

            return response, response_time

    def instructions(self, text, max_wait=2):

        """
        Shows instruction text

        Args:
            text: Text to display
            max_wait: The maximum amount of time to wait for a response before moving on

        Returns:

        """

        # set text
        self.instruction_text.text = text
        # draw
        self.instruction_text.draw()
        if max_wait > 0:
            self.win.flip()
        # waitkeys
        event.waitKeys(maxWait=max_wait, keyList=['space', '1'])

    def main_instructions(self, text_file, continue_keys='a', return_keys='l'):

        """
        Displays instruction text files for main instructions, training instructions, task instructions

        Args:
            text_file: A list of strings
            continue_keys: Keys to continue task
            return_key: Used to return to something else

        Returns:
            True if continuing, False if returning

        """

        if continue_keys is not None:
            continue_keys = ['escape', 'esc'] + list(continue_keys)

        if return_keys is not None:
            return_keys = list(return_keys)
        else:
            return_keys = []

        if not isinstance(text_file, list):
            raise TypeError("Text input is not a list")

        for i in text_file:
            self.instruction_text.text = i
            self.instruction_text.draw()
            self.win.flip()
            core.wait(1)
            key = event.waitKeys(keyList=continue_keys + return_keys)
            if key[0] in ['escape', 'esc']:
                core.quit()
            elif key[0] in continue_keys:
                return True
            elif key[0] in return_keys:
                print("Returning")
                return False


    def trigger_shock(self, voltage):

        """
        Parameters
        ----------
        voltage: Voltage in mA

        """

        if voltage < 0:
            voltage = 0

        if voltage / 100 > self.abs_max_voltage:
            print("!!! WARNING !!!!\n" \
                  "Requested voltage exceeds specified maximum " \
                  "voltage, setting to {0} degrees".format(self.abs_max_voltage))

        self.analogOutputTask.start()
        self.analogOutputTask.write(voltage / 100, auto_start=False)
        self.analogOutputTask.stop()


class CheckerBoard(object):

    """
    Adapted from https://discourse.psychopy.org/t/changing-colors-of-tiles-in-a-grid-psychopy-help/4616
    """

    def __init__(self, win, length=21, size=3, max_alpha=0.5):
        self.win = win

        location = [0, 0]
        # generate loc array
        loc = np.array(location) + np.array((size, size)) // 2

        # array of rgbs for each element (2D)
        colors = np.ones((length ** 2, 3))
        colors[::2] = 0

        # array of coordinates for each element
        xys = []
        # populate xys
        low, high = length // -2, length // 2

        for y in range(low, high):
            for x in range(low, high):
                xys.append((size * x,
                            size * y))

        self.stim = visual.ElementArrayStim(win,
                                            xys=xys,
                                            fieldPos=loc,
                                            colors=colors,
                                            nElements=length ** 2,
                                            elementMask=None,
                                            elementTex=None,
                                            opacities=max_alpha,
                                            sizes=(size, size))

        self.stim.size = (size * length,
                     size * length)

    def draw(self):
        self.stim.draw()


class ConditioningTrial(object):

    def __init__(self, task, trial_number, stimulation=True, max_voltage=20):

        """
        Conditioning trial class

        Parameters
        ----------
        task: Instance of the task class
        trial_number: Trial number
        stimulation: Indicates whether stimulation is given - can be used to turn off stimulation
        max_voltage: Voltage for stimulation

        """

        self.task = task
        self.trial_number = trial_number
        self.stimulation = stimulation
        self.max_voltage = max_voltage
        self.checkerboard = CheckerBoard(self.task.win, max_alpha=self.task.config['stimuli']['checkerboard_max_alpha'])
        self.pain_response = None
        self.hold = self.task.config['durations']['hold_time']
        self.fixation = self.task.config['durations']['initial_fixation_time']


        self.confidence_rating = None

        self.task.keyboard.clearEvents()



    def stimuli(self, t, catch, start=0, max_voltage=50, start_volt=0, end_volt=None):

        """
        Parameters
        ----------
        t: Current time
        catch: Whether this is a catch trial (bool)
        start: Time when trial starts
        max_voltage: Maximum voltage administered in the trial - can be negative


        """

        # Always show fixation
        self.task.fixation.draw()

        # Trigger Trial
        if start < t < start + self.hold:

            if catch:
                highA = sound.Sound('A', octave=3, secs=0.3, stereo=False)
                highA.setVolume(1)
                highA.play()
                self.task.question_mark.opacity = 1
                self.task.question_mark.text = ''
                self.task.question_mark.text = '?'
                self.task.question_mark.draw()

            else:   

                self.task.trigger_shock(max_voltage)
                self.checkerboard.stim.opacities = self.task.config['stimuli']['checkerboard_max_alpha']
                self.checkerboard.draw()

        # Shut down
        elif start + self.hold <= t:

            if catch:
                self.task.question_mark.opacity = 0
                self.task.question_mark.text = ''
                self.task.question_mark.text = '?'
                self.task.question_mark.draw()
            else:
                self.checkerboard.stim.opacities = 0 * self.task.config['stimuli']['checkerboard_max_alpha']
                self.checkerboard.draw()

            self.task.trigger_shock(max_voltage - max_voltage)
            self.task.keyboard.clearEvents()  # Ignore responses made in this period

    def save_data(self, save_path, session=0, block=0, stimulation_level=0, catch=False):

        self.task.data['trial_number'].append(self.trial_number)
        self.task.data['session'].append(session)
        self.task.data['block'].append(block)
        self.task.data['stimulation_level'].append(stimulation_level)
        self.task.data['voltage'].append(self.max_voltage)
        self.task.data['response'].append(self.pain_response)
        self.task.data['detected'].append(self.detected)
        self.task.data['confidence'].append(self.confidence_rating)
        self.task.data['catch'].append(catch)

        df = pd.DataFrame(self.task.data)

        df.to_csv(save_path, index=False)

    def run(self, confidence=False, end_on_response=True, catch=False, save_data=True):

        """
        Runs the trial

        Parameters
        ----------
        confidence: Indicates whether this is a confidence trial
        end_on_response: Ends the trial once they've made a response
        catch: Indicates whether this is a catch trial
        save_data: Whether to save the data

        Returns
        -------
        Subject's response - correct or incorrect

        """

        # Reset the clock
        self.task.clock.reset()

        continue_trial = True
        key_pressed = None
        detected = None

        self.task.trigger_shock(0)

        # RUN THE TRIAL
        self.task.analogOutputTask.stop()

        while continue_trial:

            t = self.task.clock.getTime()  # get the time

            # Present stimuli
            if t < self.fixation:
                
            self.stimuli(t, catch, start=self.fixation, max_voltage=self.max_voltage)

            # Get response
            if self.fixation + self.hold <= t < \
                    self.fixation + self.hold + \
                    self.task.config['durations']['response_time']:  # if we're after the stimulus presentation

                # Get keybaord events
                keys = self.task.keyboard.getEvents()

                # If no key is pressed yet, show fixation cross
                if not len(keys) and key_pressed is None:
                    self.task.fixation.draw()

                # Otherwise, show the confidence rating scale or just get responses and move on for binary trials
                else:

                    # Draw the confidence scale if we're on a confidence trial
                    if confidence:
                        self.task.confidence_scale.draw()

                    # Otherwise just show a fixation cross
                    else:
                        self.task.fixation.draw()

                    # Deal with keyboard presses
                    for key_event in keys:

                        # If the subject pressed a valid key
                        if key_event.type == EventConstants.KEYBOARD_PRESS and \
                                key_event.key in [self.task.pain_key, self.task.no_pain_key]:
                            key_pressed = (key_event.key, key_event.time)

                            # Get response
                            if key_pressed[0] == self.task.pain_key:
                                self.pain_response = True
                            elif key_pressed[0] == self.task.no_pain_key:
                                self.pain_response = False

                            # End the trial if we're not getting confidence ratings
                            if not confidence and end_on_response:
                                self.task.win.flip()
                                core.wait(0.1)
                                continue_trial = False

                        # Once they release the key, end the trial after a short delay
                        elif key_event.type == EventConstants.KEYBOARD_RELEASE:

                            if end_on_response:
                                self.task.win.flip()
                                core.wait(0.1)
                                continue_trial = False

                    # Reset the confidence scale
                    if key_pressed is not None and confidence:
                        self.task.confidence_scale.fill_bars(int(np.floor(core.getTime() - key_pressed[1])))

            # flip to draw everything
            self.task.win.flip()

            # End trial
            if t > self.fixation + self.hold + self.task.config['durations']['response_time']:
                continue_trial = False

            # If the trial has ended
            if not continue_trial:
                if confidence:
                    self.confidence_rating = self.task.confidence_scale.confidence_rating

                if self.pain_response == True and self.stimulation == True:
                    self.detected = True
                else:
                    self.detected = False
                if confidence:
                    self.task.confidence_scale.fill_bars(0)
                print("Trial done")

            # quit if subject pressed scape
            if event.getKeys(["escape", "esc"]):
                core.quit()

        return self.detected


class ConfidencePracticeTrial(ConditioningTrial):

    def __init__(self, task):
        super(ConfidencePracticeTrial, self).__init__(task, trial_number=0, stimulation=False, max_voltage=None)

        self.task = task
        self.checkerboard = None
        self.pain_response = None
        self.hold = 0
        self.fixation = 0


        self.task.keyboard.clearEvents()

def trigger(port, data):

    port.setData(data)

## RUN THE EXPERIMENT

# Create experiment class
task = PainConditoningTask('pain_hallucination_task_settings.yaml')

# Run the experiment
task.run()
task.analogOutputTask.stop()

