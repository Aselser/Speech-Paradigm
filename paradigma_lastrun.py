#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on agosto 19, 2024, at 12:18
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '2'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from code_general
from audioRecorder import AudioRecorder
import pandas as pd
import os
# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'paradigma'  # from the Builder filename that created this script
expInfo = {
    'Nombre': '',
    'Edad': '',
    'Lateralidad': ["","Derecha","Izquierda"],
    'COM': list(range(1,25)),
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s/%s' % (expInfo['Nombre'], expName)
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\agust\\OneDrive\\Desktop\\CNC\\Paradigmas\\Python\\speech-paradigm\\paradigma_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1536, 864], fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "pulses_trial_instructions" ---
    pulses_instructions = visual.TextStim(win=win, name='pulses_instructions',
        text='A continuacion se mandaran 5 pulsos al equipo, acompañados de un sonido espaciados con 2 segundos. \n\nPulse ESPACIO y espere 2 segundos.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "blank_2" ---
    blank = visual.TextStim(win=win, name='blank',
        text=' ',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "pulses" ---
    sound_1 = sound.Sound('A', secs=0.5, stereo=True, hamming=True,
        name='sound_1')
    sound_1.setVolume(1.0)
    
    # --- Initialize components for Routine "pulses_response" ---
    text = visual.TextStim(win=win, name='text',
        text='Si vio los 5 pulsos en la señal, presione ESPACIO para continuar.\n\nSi no, presione ESC y DESCONECTE el dispositivo serial.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_5 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "welcome" ---
    text_welcome = visual.TextStim(win=win, name='text_welcome',
        text=open(f'resources/TextosInstrucciones/0.Bienvenida.txt', encoding='utf-8').read()
    ,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_welcome = keyboard.Keyboard()
    # Run 'Begin Experiment' code from code_general
    puerto = 'COM' + expInfo['COM']
    recorder = AudioRecorder(mic_id = 0, sample_rate = 48000, channels = 1, arduino_port = puerto)
    
    # --- Initialize components for Routine "typical_day_instructions" ---
    text_typical_day_instructions = visual.TextStim(win=win, name='text_typical_day_instructions',
        text=open(f'resources/TextosInstrucciones/1.DiaTipico.txt', encoding='utf-8').read()
    ,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_typical_day_instructions = keyboard.Keyboard()
    
    # --- Initialize components for Routine "typical_day_recording" ---
    key_resp_typical_day = keyboard.Keyboard()
    image_microphone_typical_day = visual.ImageStim(
        win=win,
        name='image_microphone_typical_day', 
        image='resources/microphone.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "pleasant_memory_instructions" ---
    text_pleasant_memory_instructions = visual.TextStim(win=win, name='text_pleasant_memory_instructions',
        text=open(f'resources/TextosInstrucciones/2.RecuerdoAgradable.txt', encoding='utf-8').read()
    ,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_pleasant_memory_instructions = keyboard.Keyboard()
    
    # --- Initialize components for Routine "pleasant_memory_recording" ---
    key_resp_pleasant_memory = keyboard.Keyboard()
    image_microphone_pleasant_memory = visual.ImageStim(
        win=win,
        name='image_microphone_pleasant_memory', 
        image='resources/microphone.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "picture_description_instructions" ---
    text_picture_decription_instructions = visual.TextStim(win=win, name='text_picture_decription_instructions',
        text=open(f'resources/TextosInstrucciones/3.DescripcionDeLamina1.txt', encoding='utf-8').read()
    ,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_picture_description_instructions = keyboard.Keyboard()
    
    # --- Initialize components for Routine "picture" ---
    picture_descrition_image = visual.ImageStim(
        win=win,
        name='picture_descrition_image', 
        image="resources/DescripcionDeLamina1.png", mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=[0.8],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_resp_picture_description = keyboard.Keyboard()
    
    # --- Initialize components for Routine "picture_description_instructions_2" ---
    text_picture_decription_instructions_2 = visual.TextStim(win=win, name='text_picture_decription_instructions_2',
        text=open(f'resources/TextosInstrucciones/3.DescripcionDeLamina1.txt', encoding='utf-8').read()
    ,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_picture_description_instructions_2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "picture_2" ---
    picture_descrition_image_2 = visual.ImageStim(
        win=win,
        name='picture_descrition_image_2', 
        image=f"resources/DescripcionDeLamina2.jpg", mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=[0.8],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_resp_picture_description_2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "retelling_instructions" ---
    text_retelling_instructions = visual.TextStim(win=win, name='text_retelling_instructions',
        text=open(f'resources/TextosInstrucciones/5.a.RenarracionDeHistoria.txt', encoding='utf-8').read()
    ,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_retelling = keyboard.Keyboard()
    
    # --- Initialize components for Routine "story" ---
    story_retelling = sound.Sound('resources/Audio_re-narración.wav', secs=-1, stereo=True, hamming=True,
        name='story_retelling')
    story_retelling.setVolume(1.0)
    
    # --- Initialize components for Routine "retelling_instructions_2" ---
    text_retelling_instructions_2 = visual.TextStim(win=win, name='text_retelling_instructions_2',
        text=open(f'resources/TextosInstrucciones/5.b.RenarrancionDeHistoria.txt', encoding='utf-8').read()
    ,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_retelling_2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "retelling_recording" ---
    key_resp_retelling_recording = keyboard.Keyboard()
    image_microphone_retelling = visual.ImageStim(
        win=win,
        name='image_microphone_retelling', 
        image='resources/microphone.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "reading_instructions" ---
    text_reading_instructions = visual.TextStim(win=win, name='text_reading_instructions',
        text=open(f'resources/TextosInstrucciones/6.LecturaDeParrafo.txt', encoding='utf-8').read()
    ,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_reading_instructions = keyboard.Keyboard()
    
    # --- Initialize components for Routine "paragraph" ---
    key_resp_reading = keyboard.Keyboard()
    paragraph_image = visual.ImageStim(
        win=win,
        name='paragraph_image', 
        image='resources/texto.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(1.75, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "letter_A_instructions" ---
    text_letter_A_instructions = visual.TextStim(win=win, name='text_letter_A_instructions',
        text=open(f'resources/TextosInstrucciones/7.VocalA.txt', encoding='utf-8').read()
    ,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_letter_A_instructions = keyboard.Keyboard()
    
    # --- Initialize components for Routine "letter_A_recording" ---
    key_resp_letter_A = keyboard.Keyboard()
    image_microphone_letter_A = visual.ImageStim(
        win=win,
        name='image_microphone_letter_A', 
        image='resources/microphone.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "pataka_instructions" ---
    text_pataka_instructions = visual.TextStim(win=win, name='text_pataka_instructions',
        text=open('resources/TextosInstrucciones/8.Pataka.txt', encoding='utf-8').read()
    ,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_pataka_instructions = keyboard.Keyboard()
    
    # --- Initialize components for Routine "pataka_recording" ---
    key_resp_pataka = keyboard.Keyboard()
    image_microphone_pataka = visual.ImageStim(
        win=win,
        name='image_microphone_pataka', 
        image='resources/microphone.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "acknowledgment" ---
    text_end = visual.TextStim(win=win, name='text_end',
        text=open('resources/TextosInstrucciones/9.CuestionarioYFin.txt', encoding='utf-8').read()
    ,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_end = keyboard.Keyboard()
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "pulses_trial_instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # keep track of which components have finished
    pulses_trial_instructionsComponents = [pulses_instructions, key_resp]
    for thisComponent in pulses_trial_instructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "pulses_trial_instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *pulses_instructions* updates
        
        # if pulses_instructions is starting this frame...
        if pulses_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            pulses_instructions.frameNStart = frameN  # exact frame index
            pulses_instructions.tStart = t  # local t and not account for scr refresh
            pulses_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(pulses_instructions, 'tStartRefresh')  # time at next scr refresh
            # update status
            pulses_instructions.status = STARTED
            pulses_instructions.setAutoDraw(True)
        
        # if pulses_instructions is active this frame...
        if pulses_instructions.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in pulses_trial_instructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "pulses_trial_instructions" ---
    for thisComponent in pulses_trial_instructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "pulses_trial_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=5.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "blank_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        # keep track of which components have finished
        blank_2Components = [blank]
        for thisComponent in blank_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *blank* updates
            
            # if blank is starting this frame...
            if blank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                blank.frameNStart = frameN  # exact frame index
                blank.tStart = t  # local t and not account for scr refresh
                blank.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blank, 'tStartRefresh')  # time at next scr refresh
                # update status
                blank.status = STARTED
                blank.setAutoDraw(True)
            
            # if blank is active this frame...
            if blank.status == STARTED:
                # update params
                pass
            
            # if blank is stopping this frame...
            if blank.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blank.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    blank.tStop = t  # not accounting for scr refresh
                    blank.frameNStop = frameN  # exact frame index
                    # update status
                    blank.status = FINISHED
                    blank.setAutoDraw(False)
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank_2" ---
        for thisComponent in blank_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "pulses" ---
        continueRoutine = True
        # update component parameters for each repeat
        sound_1.setSound('A', secs=0.5, hamming=True)
        sound_1.setVolume(1.0, log=False)
        sound_1.seek(0)
        # Run 'Begin Routine' code from code_2
        recorder._send_pulse_to_arduino()
        # keep track of which components have finished
        pulsesComponents = [sound_1]
        for thisComponent in pulsesComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "pulses" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # if sound_1 is starting this frame...
            if sound_1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                sound_1.frameNStart = frameN  # exact frame index
                sound_1.tStart = t  # local t and not account for scr refresh
                sound_1.tStartRefresh = tThisFlipGlobal  # on global time
                # update status
                sound_1.status = STARTED
                sound_1.play(when=win)  # sync with win flip
            
            # if sound_1 is stopping this frame...
            if sound_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_1.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_1.tStop = t  # not accounting for scr refresh
                    sound_1.frameNStop = frameN  # exact frame index
                    # update status
                    sound_1.status = FINISHED
                    sound_1.stop()
            # update sound_1 status according to whether it's playing
            if sound_1.isPlaying:
                sound_1.status = STARTED
            elif sound_1.isFinished:
                sound_1.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in pulsesComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pulses" ---
        for thisComponent in pulsesComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        sound_1.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
    # completed 5.0 repeats of 'trials'
    
    
    # --- Prepare to start Routine "pulses_response" ---
    continueRoutine = True
    # update component parameters for each repeat
    key_resp_5.keys = []
    key_resp_5.rt = []
    _key_resp_5_allKeys = []
    # keep track of which components have finished
    pulses_responseComponents = [text, key_resp_5]
    for thisComponent in pulses_responseComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "pulses_response" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *key_resp_5* updates
        waitOnFlip = False
        
        # if key_resp_5 is starting this frame...
        if key_resp_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_5.frameNStart = frameN  # exact frame index
            key_resp_5.tStart = t  # local t and not account for scr refresh
            key_resp_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_5, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_5.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_5.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_5.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_5.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _key_resp_5_allKeys.extend(theseKeys)
            if len(_key_resp_5_allKeys):
                key_resp_5.keys = _key_resp_5_allKeys[-1].name  # just the last key pressed
                key_resp_5.rt = _key_resp_5_allKeys[-1].rt
                key_resp_5.duration = _key_resp_5_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in pulses_responseComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "pulses_response" ---
    for thisComponent in pulses_responseComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "pulses_response" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "welcome" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('welcome.started', globalClock.getTime())
    key_resp_welcome.keys = []
    key_resp_welcome.rt = []
    _key_resp_welcome_allKeys = []
    # keep track of which components have finished
    welcomeComponents = [text_welcome, key_resp_welcome]
    for thisComponent in welcomeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "welcome" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_welcome* updates
        
        # if text_welcome is starting this frame...
        if text_welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_welcome.frameNStart = frameN  # exact frame index
            text_welcome.tStart = t  # local t and not account for scr refresh
            text_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_welcome, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_welcome.status = STARTED
            text_welcome.setAutoDraw(True)
        
        # if text_welcome is active this frame...
        if text_welcome.status == STARTED:
            # update params
            pass
        
        # *key_resp_welcome* updates
        waitOnFlip = False
        
        # if key_resp_welcome is starting this frame...
        if key_resp_welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_welcome.frameNStart = frameN  # exact frame index
            key_resp_welcome.tStart = t  # local t and not account for scr refresh
            key_resp_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_welcome, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_welcome.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_welcome.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_welcome.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_welcome.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_welcome.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _key_resp_welcome_allKeys.extend(theseKeys)
            if len(_key_resp_welcome_allKeys):
                key_resp_welcome.keys = _key_resp_welcome_allKeys[-1].name  # just the last key pressed
                key_resp_welcome.rt = _key_resp_welcome_allKeys[-1].rt
                key_resp_welcome.duration = _key_resp_welcome_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcomeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome" ---
    for thisComponent in welcomeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('welcome.stopped', globalClock.getTime())
    # check responses
    if key_resp_welcome.keys in ['', [], None]:  # No response was made
        key_resp_welcome.keys = None
    thisExp.addData('key_resp_welcome.keys',key_resp_welcome.keys)
    if key_resp_welcome.keys != None:  # we had a response
        thisExp.addData('key_resp_welcome.rt', key_resp_welcome.rt)
        thisExp.addData('key_resp_welcome.duration', key_resp_welcome.duration)
    thisExp.nextEntry()
    # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "typical_day_instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('typical_day_instructions.started', globalClock.getTime())
    key_resp_typical_day_instructions.keys = []
    key_resp_typical_day_instructions.rt = []
    _key_resp_typical_day_instructions_allKeys = []
    # keep track of which components have finished
    typical_day_instructionsComponents = [text_typical_day_instructions, key_resp_typical_day_instructions]
    for thisComponent in typical_day_instructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "typical_day_instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_typical_day_instructions* updates
        
        # if text_typical_day_instructions is starting this frame...
        if text_typical_day_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_typical_day_instructions.frameNStart = frameN  # exact frame index
            text_typical_day_instructions.tStart = t  # local t and not account for scr refresh
            text_typical_day_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_typical_day_instructions, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_typical_day_instructions.status = STARTED
            text_typical_day_instructions.setAutoDraw(True)
        
        # if text_typical_day_instructions is active this frame...
        if text_typical_day_instructions.status == STARTED:
            # update params
            pass
        
        # *key_resp_typical_day_instructions* updates
        waitOnFlip = False
        
        # if key_resp_typical_day_instructions is starting this frame...
        if key_resp_typical_day_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_typical_day_instructions.frameNStart = frameN  # exact frame index
            key_resp_typical_day_instructions.tStart = t  # local t and not account for scr refresh
            key_resp_typical_day_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_typical_day_instructions, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_typical_day_instructions.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_typical_day_instructions.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_typical_day_instructions.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_typical_day_instructions.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_typical_day_instructions.getKeys(keyList=['space','s'], ignoreKeys=None, waitRelease=False)
            _key_resp_typical_day_instructions_allKeys.extend(theseKeys)
            if len(_key_resp_typical_day_instructions_allKeys):
                key_resp_typical_day_instructions.keys = _key_resp_typical_day_instructions_allKeys[-1].name  # just the last key pressed
                key_resp_typical_day_instructions.rt = _key_resp_typical_day_instructions_allKeys[-1].rt
                key_resp_typical_day_instructions.duration = _key_resp_typical_day_instructions_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in typical_day_instructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "typical_day_instructions" ---
    for thisComponent in typical_day_instructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('typical_day_instructions.stopped', globalClock.getTime())
    # check responses
    if key_resp_typical_day_instructions.keys in ['', [], None]:  # No response was made
        key_resp_typical_day_instructions.keys = None
    thisExp.addData('key_resp_typical_day_instructions.keys',key_resp_typical_day_instructions.keys)
    if key_resp_typical_day_instructions.keys != None:  # we had a response
        thisExp.addData('key_resp_typical_day_instructions.rt', key_resp_typical_day_instructions.rt)
        thisExp.addData('key_resp_typical_day_instructions.duration', key_resp_typical_day_instructions.duration)
    thisExp.nextEntry()
    # the Routine "typical_day_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "typical_day_recording" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('typical_day_recording.started', globalClock.getTime())
    # skip this Routine if its 'Skip if' condition is True
    continueRoutine = continueRoutine and not (key_resp_typical_day_instructions.keys == 's')
    key_resp_typical_day.keys = []
    key_resp_typical_day.rt = []
    _key_resp_typical_day_allKeys = []
    # Run 'Begin Routine' code from code_6
    if key_resp_typical_day_instructions.keys != 's':
        recorder.start_recording(f'data/{expInfo["Nombre"]}/{expInfo["Nombre"]}_DiaTipico.wav')
    # keep track of which components have finished
    typical_day_recordingComponents = [key_resp_typical_day, image_microphone_typical_day]
    for thisComponent in typical_day_recordingComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "typical_day_recording" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_resp_typical_day* updates
        waitOnFlip = False
        
        # if key_resp_typical_day is starting this frame...
        if key_resp_typical_day.status == NOT_STARTED and tThisFlip >= 5.5-frameTolerance:
            # keep track of start time/frame for later
            key_resp_typical_day.frameNStart = frameN  # exact frame index
            key_resp_typical_day.tStart = t  # local t and not account for scr refresh
            key_resp_typical_day.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_typical_day, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_typical_day.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_typical_day.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_typical_day.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_typical_day.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_typical_day.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _key_resp_typical_day_allKeys.extend(theseKeys)
            if len(_key_resp_typical_day_allKeys):
                key_resp_typical_day.keys = _key_resp_typical_day_allKeys[-1].name  # just the last key pressed
                key_resp_typical_day.rt = _key_resp_typical_day_allKeys[-1].rt
                key_resp_typical_day.duration = _key_resp_typical_day_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *image_microphone_typical_day* updates
        
        # if image_microphone_typical_day is starting this frame...
        if image_microphone_typical_day.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            image_microphone_typical_day.frameNStart = frameN  # exact frame index
            image_microphone_typical_day.tStart = t  # local t and not account for scr refresh
            image_microphone_typical_day.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_microphone_typical_day, 'tStartRefresh')  # time at next scr refresh
            # update status
            image_microphone_typical_day.status = STARTED
            image_microphone_typical_day.setAutoDraw(True)
        
        # if image_microphone_typical_day is active this frame...
        if image_microphone_typical_day.status == STARTED:
            # update params
            pass
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in typical_day_recordingComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "typical_day_recording" ---
    for thisComponent in typical_day_recordingComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('typical_day_recording.stopped', globalClock.getTime())
    # Run 'End Routine' code from code_6
    if key_resp_typical_day_instructions.keys != 's':
        recorder.stop_recording()
    # the Routine "typical_day_recording" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "pleasant_memory_instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('pleasant_memory_instructions.started', globalClock.getTime())
    key_resp_pleasant_memory_instructions.keys = []
    key_resp_pleasant_memory_instructions.rt = []
    _key_resp_pleasant_memory_instructions_allKeys = []
    # keep track of which components have finished
    pleasant_memory_instructionsComponents = [text_pleasant_memory_instructions, key_resp_pleasant_memory_instructions]
    for thisComponent in pleasant_memory_instructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "pleasant_memory_instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_pleasant_memory_instructions* updates
        
        # if text_pleasant_memory_instructions is starting this frame...
        if text_pleasant_memory_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_pleasant_memory_instructions.frameNStart = frameN  # exact frame index
            text_pleasant_memory_instructions.tStart = t  # local t and not account for scr refresh
            text_pleasant_memory_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_pleasant_memory_instructions, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_pleasant_memory_instructions.status = STARTED
            text_pleasant_memory_instructions.setAutoDraw(True)
        
        # if text_pleasant_memory_instructions is active this frame...
        if text_pleasant_memory_instructions.status == STARTED:
            # update params
            pass
        
        # *key_resp_pleasant_memory_instructions* updates
        waitOnFlip = False
        
        # if key_resp_pleasant_memory_instructions is starting this frame...
        if key_resp_pleasant_memory_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_pleasant_memory_instructions.frameNStart = frameN  # exact frame index
            key_resp_pleasant_memory_instructions.tStart = t  # local t and not account for scr refresh
            key_resp_pleasant_memory_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_pleasant_memory_instructions, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_pleasant_memory_instructions.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_pleasant_memory_instructions.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_pleasant_memory_instructions.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_pleasant_memory_instructions.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_pleasant_memory_instructions.getKeys(keyList=['space','s'], ignoreKeys=None, waitRelease=False)
            _key_resp_pleasant_memory_instructions_allKeys.extend(theseKeys)
            if len(_key_resp_pleasant_memory_instructions_allKeys):
                key_resp_pleasant_memory_instructions.keys = _key_resp_pleasant_memory_instructions_allKeys[-1].name  # just the last key pressed
                key_resp_pleasant_memory_instructions.rt = _key_resp_pleasant_memory_instructions_allKeys[-1].rt
                key_resp_pleasant_memory_instructions.duration = _key_resp_pleasant_memory_instructions_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in pleasant_memory_instructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "pleasant_memory_instructions" ---
    for thisComponent in pleasant_memory_instructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('pleasant_memory_instructions.stopped', globalClock.getTime())
    # check responses
    if key_resp_pleasant_memory_instructions.keys in ['', [], None]:  # No response was made
        key_resp_pleasant_memory_instructions.keys = None
    thisExp.addData('key_resp_pleasant_memory_instructions.keys',key_resp_pleasant_memory_instructions.keys)
    if key_resp_pleasant_memory_instructions.keys != None:  # we had a response
        thisExp.addData('key_resp_pleasant_memory_instructions.rt', key_resp_pleasant_memory_instructions.rt)
        thisExp.addData('key_resp_pleasant_memory_instructions.duration', key_resp_pleasant_memory_instructions.duration)
    thisExp.nextEntry()
    # the Routine "pleasant_memory_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "pleasant_memory_recording" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('pleasant_memory_recording.started', globalClock.getTime())
    # skip this Routine if its 'Skip if' condition is True
    continueRoutine = continueRoutine and not (key_resp_pleasant_memory_instructions.keys == 's')
    key_resp_pleasant_memory.keys = []
    key_resp_pleasant_memory.rt = []
    _key_resp_pleasant_memory_allKeys = []
    # Run 'Begin Routine' code from code_7
    if key_resp_pleasant_memory_instructions.keys != 's':
        recorder.start_recording(f'data/{expInfo["Nombre"]}/{expInfo["Nombre"]}_RecuerdoAgradable.wav')
    # keep track of which components have finished
    pleasant_memory_recordingComponents = [key_resp_pleasant_memory, image_microphone_pleasant_memory]
    for thisComponent in pleasant_memory_recordingComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "pleasant_memory_recording" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_resp_pleasant_memory* updates
        waitOnFlip = False
        
        # if key_resp_pleasant_memory is starting this frame...
        if key_resp_pleasant_memory.status == NOT_STARTED and tThisFlip >= 5.5-frameTolerance:
            # keep track of start time/frame for later
            key_resp_pleasant_memory.frameNStart = frameN  # exact frame index
            key_resp_pleasant_memory.tStart = t  # local t and not account for scr refresh
            key_resp_pleasant_memory.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_pleasant_memory, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_pleasant_memory.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_pleasant_memory.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_pleasant_memory.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_pleasant_memory.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_pleasant_memory.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _key_resp_pleasant_memory_allKeys.extend(theseKeys)
            if len(_key_resp_pleasant_memory_allKeys):
                key_resp_pleasant_memory.keys = _key_resp_pleasant_memory_allKeys[-1].name  # just the last key pressed
                key_resp_pleasant_memory.rt = _key_resp_pleasant_memory_allKeys[-1].rt
                key_resp_pleasant_memory.duration = _key_resp_pleasant_memory_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *image_microphone_pleasant_memory* updates
        
        # if image_microphone_pleasant_memory is starting this frame...
        if image_microphone_pleasant_memory.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            image_microphone_pleasant_memory.frameNStart = frameN  # exact frame index
            image_microphone_pleasant_memory.tStart = t  # local t and not account for scr refresh
            image_microphone_pleasant_memory.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_microphone_pleasant_memory, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_microphone_pleasant_memory.started')
            # update status
            image_microphone_pleasant_memory.status = STARTED
            image_microphone_pleasant_memory.setAutoDraw(True)
        
        # if image_microphone_pleasant_memory is active this frame...
        if image_microphone_pleasant_memory.status == STARTED:
            # update params
            pass
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in pleasant_memory_recordingComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "pleasant_memory_recording" ---
    for thisComponent in pleasant_memory_recordingComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('pleasant_memory_recording.stopped', globalClock.getTime())
    # Run 'End Routine' code from code_7
    if key_resp_pleasant_memory_instructions.keys != 's':
        recorder.stop_recording()
    # the Routine "pleasant_memory_recording" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "picture_description_instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('picture_description_instructions.started', globalClock.getTime())
    key_resp_picture_description_instructions.keys = []
    key_resp_picture_description_instructions.rt = []
    _key_resp_picture_description_instructions_allKeys = []
    # keep track of which components have finished
    picture_description_instructionsComponents = [text_picture_decription_instructions, key_resp_picture_description_instructions]
    for thisComponent in picture_description_instructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "picture_description_instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_picture_decription_instructions* updates
        
        # if text_picture_decription_instructions is starting this frame...
        if text_picture_decription_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_picture_decription_instructions.frameNStart = frameN  # exact frame index
            text_picture_decription_instructions.tStart = t  # local t and not account for scr refresh
            text_picture_decription_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_picture_decription_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_picture_decription_instructions.started')
            # update status
            text_picture_decription_instructions.status = STARTED
            text_picture_decription_instructions.setAutoDraw(True)
        
        # if text_picture_decription_instructions is active this frame...
        if text_picture_decription_instructions.status == STARTED:
            # update params
            pass
        
        # *key_resp_picture_description_instructions* updates
        waitOnFlip = False
        
        # if key_resp_picture_description_instructions is starting this frame...
        if key_resp_picture_description_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_picture_description_instructions.frameNStart = frameN  # exact frame index
            key_resp_picture_description_instructions.tStart = t  # local t and not account for scr refresh
            key_resp_picture_description_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_picture_description_instructions, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_picture_description_instructions.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_picture_description_instructions.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_picture_description_instructions.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_picture_description_instructions.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_picture_description_instructions.getKeys(keyList=['space','s'], ignoreKeys=None, waitRelease=False)
            _key_resp_picture_description_instructions_allKeys.extend(theseKeys)
            if len(_key_resp_picture_description_instructions_allKeys):
                key_resp_picture_description_instructions.keys = _key_resp_picture_description_instructions_allKeys[-1].name  # just the last key pressed
                key_resp_picture_description_instructions.rt = _key_resp_picture_description_instructions_allKeys[-1].rt
                key_resp_picture_description_instructions.duration = _key_resp_picture_description_instructions_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in picture_description_instructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "picture_description_instructions" ---
    for thisComponent in picture_description_instructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('picture_description_instructions.stopped', globalClock.getTime())
    # the Routine "picture_description_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "picture" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('picture.started', globalClock.getTime())
    # skip this Routine if its 'Skip if' condition is True
    continueRoutine = continueRoutine and not (key_resp_picture_description_instructions.keys == 's')
    key_resp_picture_description.keys = []
    key_resp_picture_description.rt = []
    _key_resp_picture_description_allKeys = []
    # Run 'Begin Routine' code from code_3
    if key_resp_picture_description_instructions.keys != 's': 
        recorder.start_recording(f'data/{expInfo["Nombre"]}/{expInfo["Nombre"]}_DescripcionDeLamina1.wav')
    # keep track of which components have finished
    pictureComponents = [picture_descrition_image, key_resp_picture_description]
    for thisComponent in pictureComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "picture" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *picture_descrition_image* updates
        
        # if picture_descrition_image is starting this frame...
        if picture_descrition_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            picture_descrition_image.frameNStart = frameN  # exact frame index
            picture_descrition_image.tStart = t  # local t and not account for scr refresh
            picture_descrition_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(picture_descrition_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'picture_descrition_image.started')
            # update status
            picture_descrition_image.status = STARTED
            picture_descrition_image.setAutoDraw(True)
        
        # if picture_descrition_image is active this frame...
        if picture_descrition_image.status == STARTED:
            # update params
            pass
        
        # *key_resp_picture_description* updates
        waitOnFlip = False
        
        # if key_resp_picture_description is starting this frame...
        if key_resp_picture_description.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_picture_description.frameNStart = frameN  # exact frame index
            key_resp_picture_description.tStart = t  # local t and not account for scr refresh
            key_resp_picture_description.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_picture_description, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_picture_description.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_picture_description.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_picture_description.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_picture_description.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_picture_description.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _key_resp_picture_description_allKeys.extend(theseKeys)
            if len(_key_resp_picture_description_allKeys):
                key_resp_picture_description.keys = _key_resp_picture_description_allKeys[-1].name  # just the last key pressed
                key_resp_picture_description.rt = _key_resp_picture_description_allKeys[-1].rt
                key_resp_picture_description.duration = _key_resp_picture_description_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in pictureComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "picture" ---
    for thisComponent in pictureComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('picture.stopped', globalClock.getTime())
    # Run 'End Routine' code from code_3
    if key_resp_picture_description_instructions.keys != 's': 
        recorder.stop_recording()
    # the Routine "picture" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "picture_description_instructions_2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('picture_description_instructions_2.started', globalClock.getTime())
    key_resp_picture_description_instructions_2.keys = []
    key_resp_picture_description_instructions_2.rt = []
    _key_resp_picture_description_instructions_2_allKeys = []
    # keep track of which components have finished
    picture_description_instructions_2Components = [text_picture_decription_instructions_2, key_resp_picture_description_instructions_2]
    for thisComponent in picture_description_instructions_2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "picture_description_instructions_2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_picture_decription_instructions_2* updates
        
        # if text_picture_decription_instructions_2 is starting this frame...
        if text_picture_decription_instructions_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_picture_decription_instructions_2.frameNStart = frameN  # exact frame index
            text_picture_decription_instructions_2.tStart = t  # local t and not account for scr refresh
            text_picture_decription_instructions_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_picture_decription_instructions_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_picture_decription_instructions_2.started')
            # update status
            text_picture_decription_instructions_2.status = STARTED
            text_picture_decription_instructions_2.setAutoDraw(True)
        
        # if text_picture_decription_instructions_2 is active this frame...
        if text_picture_decription_instructions_2.status == STARTED:
            # update params
            pass
        
        # *key_resp_picture_description_instructions_2* updates
        waitOnFlip = False
        
        # if key_resp_picture_description_instructions_2 is starting this frame...
        if key_resp_picture_description_instructions_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_picture_description_instructions_2.frameNStart = frameN  # exact frame index
            key_resp_picture_description_instructions_2.tStart = t  # local t and not account for scr refresh
            key_resp_picture_description_instructions_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_picture_description_instructions_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_picture_description_instructions_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_picture_description_instructions_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_picture_description_instructions_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_picture_description_instructions_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_picture_description_instructions_2.getKeys(keyList=['space','s'], ignoreKeys=None, waitRelease=False)
            _key_resp_picture_description_instructions_2_allKeys.extend(theseKeys)
            if len(_key_resp_picture_description_instructions_2_allKeys):
                key_resp_picture_description_instructions_2.keys = _key_resp_picture_description_instructions_2_allKeys[-1].name  # just the last key pressed
                key_resp_picture_description_instructions_2.rt = _key_resp_picture_description_instructions_2_allKeys[-1].rt
                key_resp_picture_description_instructions_2.duration = _key_resp_picture_description_instructions_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in picture_description_instructions_2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "picture_description_instructions_2" ---
    for thisComponent in picture_description_instructions_2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('picture_description_instructions_2.stopped', globalClock.getTime())
    # the Routine "picture_description_instructions_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "picture_2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('picture_2.started', globalClock.getTime())
    # skip this Routine if its 'Skip if' condition is True
    continueRoutine = continueRoutine and not (key_resp_picture_description_instructions_2.keys == 's')
    key_resp_picture_description_2.keys = []
    key_resp_picture_description_2.rt = []
    _key_resp_picture_description_2_allKeys = []
    # Run 'Begin Routine' code from code_8
    if key_resp_picture_description_instructions_2.keys != 's':
        recorder.start_recording(f'data/{expInfo["Nombre"]}/{expInfo["Nombre"]}_DescripcionDeLamina2.wav')
    # keep track of which components have finished
    picture_2Components = [picture_descrition_image_2, key_resp_picture_description_2]
    for thisComponent in picture_2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "picture_2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *picture_descrition_image_2* updates
        
        # if picture_descrition_image_2 is starting this frame...
        if picture_descrition_image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            picture_descrition_image_2.frameNStart = frameN  # exact frame index
            picture_descrition_image_2.tStart = t  # local t and not account for scr refresh
            picture_descrition_image_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(picture_descrition_image_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'picture_descrition_image_2.started')
            # update status
            picture_descrition_image_2.status = STARTED
            picture_descrition_image_2.setAutoDraw(True)
        
        # if picture_descrition_image_2 is active this frame...
        if picture_descrition_image_2.status == STARTED:
            # update params
            pass
        
        # *key_resp_picture_description_2* updates
        waitOnFlip = False
        
        # if key_resp_picture_description_2 is starting this frame...
        if key_resp_picture_description_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_picture_description_2.frameNStart = frameN  # exact frame index
            key_resp_picture_description_2.tStart = t  # local t and not account for scr refresh
            key_resp_picture_description_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_picture_description_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_picture_description_2.started')
            # update status
            key_resp_picture_description_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_picture_description_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_picture_description_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_picture_description_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_picture_description_2.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _key_resp_picture_description_2_allKeys.extend(theseKeys)
            if len(_key_resp_picture_description_2_allKeys):
                key_resp_picture_description_2.keys = _key_resp_picture_description_2_allKeys[-1].name  # just the last key pressed
                key_resp_picture_description_2.rt = _key_resp_picture_description_2_allKeys[-1].rt
                key_resp_picture_description_2.duration = _key_resp_picture_description_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in picture_2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "picture_2" ---
    for thisComponent in picture_2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('picture_2.stopped', globalClock.getTime())
    # check responses
    if key_resp_picture_description_2.keys in ['', [], None]:  # No response was made
        key_resp_picture_description_2.keys = None
    thisExp.addData('key_resp_picture_description_2.keys',key_resp_picture_description_2.keys)
    if key_resp_picture_description_2.keys != None:  # we had a response
        thisExp.addData('key_resp_picture_description_2.rt', key_resp_picture_description_2.rt)
        thisExp.addData('key_resp_picture_description_2.duration', key_resp_picture_description_2.duration)
    thisExp.nextEntry()
    # Run 'End Routine' code from code_8
    if key_resp_picture_description_instructions_2.keys != 's':
        recorder.stop_recording()
    # the Routine "picture_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "retelling_instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('retelling_instructions.started', globalClock.getTime())
    key_resp_retelling.keys = []
    key_resp_retelling.rt = []
    _key_resp_retelling_allKeys = []
    # keep track of which components have finished
    retelling_instructionsComponents = [text_retelling_instructions, key_resp_retelling]
    for thisComponent in retelling_instructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "retelling_instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_retelling_instructions* updates
        
        # if text_retelling_instructions is starting this frame...
        if text_retelling_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_retelling_instructions.frameNStart = frameN  # exact frame index
            text_retelling_instructions.tStart = t  # local t and not account for scr refresh
            text_retelling_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_retelling_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_retelling_instructions.started')
            # update status
            text_retelling_instructions.status = STARTED
            text_retelling_instructions.setAutoDraw(True)
        
        # if text_retelling_instructions is active this frame...
        if text_retelling_instructions.status == STARTED:
            # update params
            pass
        
        # *key_resp_retelling* updates
        waitOnFlip = False
        
        # if key_resp_retelling is starting this frame...
        if key_resp_retelling.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_retelling.frameNStart = frameN  # exact frame index
            key_resp_retelling.tStart = t  # local t and not account for scr refresh
            key_resp_retelling.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_retelling, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_retelling.started')
            # update status
            key_resp_retelling.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_retelling.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_retelling.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_retelling.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_retelling.getKeys(keyList=['space','s'], ignoreKeys=None, waitRelease=False)
            _key_resp_retelling_allKeys.extend(theseKeys)
            if len(_key_resp_retelling_allKeys):
                key_resp_retelling.keys = _key_resp_retelling_allKeys[-1].name  # just the last key pressed
                key_resp_retelling.rt = _key_resp_retelling_allKeys[-1].rt
                key_resp_retelling.duration = _key_resp_retelling_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in retelling_instructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "retelling_instructions" ---
    for thisComponent in retelling_instructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('retelling_instructions.stopped', globalClock.getTime())
    # check responses
    if key_resp_retelling.keys in ['', [], None]:  # No response was made
        key_resp_retelling.keys = None
    thisExp.addData('key_resp_retelling.keys',key_resp_retelling.keys)
    if key_resp_retelling.keys != None:  # we had a response
        thisExp.addData('key_resp_retelling.rt', key_resp_retelling.rt)
        thisExp.addData('key_resp_retelling.duration', key_resp_retelling.duration)
    thisExp.nextEntry()
    # the Routine "retelling_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "story" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('story.started', globalClock.getTime())
    # skip this Routine if its 'Skip if' condition is True
    continueRoutine = continueRoutine and not (key_resp_retelling.keys == 's')
    # Run 'Begin Routine' code from code_4
    if key_resp_retelling.keys != 's':
        recorder._send_pulse_to_arduino()
    story_retelling.setSound('resources/Audio_re-narración.wav', secs=100, hamming=True)
    story_retelling.setVolume(1.0, log=False)
    story_retelling.seek(0)
    # keep track of which components have finished
    storyComponents = [story_retelling]
    for thisComponent in storyComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "story" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 100.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # if story_retelling is starting this frame...
        if story_retelling.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            story_retelling.frameNStart = frameN  # exact frame index
            story_retelling.tStart = t  # local t and not account for scr refresh
            story_retelling.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('story_retelling.started', tThisFlipGlobal)
            # update status
            story_retelling.status = STARTED
            story_retelling.play(when=win)  # sync with win flip
        
        # if story_retelling is stopping this frame...
        if story_retelling.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > story_retelling.tStartRefresh + 100-frameTolerance:
                # keep track of stop time/frame for later
                story_retelling.tStop = t  # not accounting for scr refresh
                story_retelling.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'story_retelling.stopped')
                # update status
                story_retelling.status = FINISHED
                story_retelling.stop()
        # update story_retelling status according to whether it's playing
        if story_retelling.isPlaying:
            story_retelling.status = STARTED
        elif story_retelling.isFinished:
            story_retelling.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in storyComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "story" ---
    for thisComponent in storyComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('story.stopped', globalClock.getTime())
    # Run 'End Routine' code from code_4
    if key_resp_retelling.keys != 's':
        recorder._send_pulse_to_arduino()
    story_retelling.pause()  # ensure sound has stopped at end of Routine
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-100.000000)
    
    # --- Prepare to start Routine "retelling_instructions_2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('retelling_instructions_2.started', globalClock.getTime())
    # skip this Routine if its 'Skip if' condition is True
    continueRoutine = continueRoutine and not (key_resp_retelling.keys == 's')
    key_resp_retelling_2.keys = []
    key_resp_retelling_2.rt = []
    _key_resp_retelling_2_allKeys = []
    # keep track of which components have finished
    retelling_instructions_2Components = [text_retelling_instructions_2, key_resp_retelling_2]
    for thisComponent in retelling_instructions_2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "retelling_instructions_2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_retelling_instructions_2* updates
        
        # if text_retelling_instructions_2 is starting this frame...
        if text_retelling_instructions_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_retelling_instructions_2.frameNStart = frameN  # exact frame index
            text_retelling_instructions_2.tStart = t  # local t and not account for scr refresh
            text_retelling_instructions_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_retelling_instructions_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_retelling_instructions_2.started')
            # update status
            text_retelling_instructions_2.status = STARTED
            text_retelling_instructions_2.setAutoDraw(True)
        
        # if text_retelling_instructions_2 is active this frame...
        if text_retelling_instructions_2.status == STARTED:
            # update params
            pass
        
        # *key_resp_retelling_2* updates
        waitOnFlip = False
        
        # if key_resp_retelling_2 is starting this frame...
        if key_resp_retelling_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_retelling_2.frameNStart = frameN  # exact frame index
            key_resp_retelling_2.tStart = t  # local t and not account for scr refresh
            key_resp_retelling_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_retelling_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_retelling_2.started')
            # update status
            key_resp_retelling_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_retelling_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_retelling_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_retelling_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_retelling_2.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _key_resp_retelling_2_allKeys.extend(theseKeys)
            if len(_key_resp_retelling_2_allKeys):
                key_resp_retelling_2.keys = _key_resp_retelling_2_allKeys[-1].name  # just the last key pressed
                key_resp_retelling_2.rt = _key_resp_retelling_2_allKeys[-1].rt
                key_resp_retelling_2.duration = _key_resp_retelling_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in retelling_instructions_2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "retelling_instructions_2" ---
    for thisComponent in retelling_instructions_2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('retelling_instructions_2.stopped', globalClock.getTime())
    # check responses
    if key_resp_retelling_2.keys in ['', [], None]:  # No response was made
        key_resp_retelling_2.keys = None
    thisExp.addData('key_resp_retelling_2.keys',key_resp_retelling_2.keys)
    if key_resp_retelling_2.keys != None:  # we had a response
        thisExp.addData('key_resp_retelling_2.rt', key_resp_retelling_2.rt)
        thisExp.addData('key_resp_retelling_2.duration', key_resp_retelling_2.duration)
    thisExp.nextEntry()
    # the Routine "retelling_instructions_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "retelling_recording" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('retelling_recording.started', globalClock.getTime())
    # skip this Routine if its 'Skip if' condition is True
    continueRoutine = continueRoutine and not (key_resp_retelling.keys == 's')
    key_resp_retelling_recording.keys = []
    key_resp_retelling_recording.rt = []
    _key_resp_retelling_recording_allKeys = []
    # Run 'Begin Routine' code from code_9
    if key_resp_retelling.keys != 's':
        recorder.start_recording(f'data/{expInfo["Nombre"]}/{expInfo["Nombre"]}_RenarracionDeHistoria.wav')
    # keep track of which components have finished
    retelling_recordingComponents = [key_resp_retelling_recording, image_microphone_retelling]
    for thisComponent in retelling_recordingComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "retelling_recording" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_resp_retelling_recording* updates
        waitOnFlip = False
        
        # if key_resp_retelling_recording is starting this frame...
        if key_resp_retelling_recording.status == NOT_STARTED and tThisFlip >= 5.5-frameTolerance:
            # keep track of start time/frame for later
            key_resp_retelling_recording.frameNStart = frameN  # exact frame index
            key_resp_retelling_recording.tStart = t  # local t and not account for scr refresh
            key_resp_retelling_recording.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_retelling_recording, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_retelling_recording.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_retelling_recording.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_retelling_recording.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_retelling_recording.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_retelling_recording.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _key_resp_retelling_recording_allKeys.extend(theseKeys)
            if len(_key_resp_retelling_recording_allKeys):
                key_resp_retelling_recording.keys = _key_resp_retelling_recording_allKeys[-1].name  # just the last key pressed
                key_resp_retelling_recording.rt = _key_resp_retelling_recording_allKeys[-1].rt
                key_resp_retelling_recording.duration = _key_resp_retelling_recording_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *image_microphone_retelling* updates
        
        # if image_microphone_retelling is starting this frame...
        if image_microphone_retelling.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            image_microphone_retelling.frameNStart = frameN  # exact frame index
            image_microphone_retelling.tStart = t  # local t and not account for scr refresh
            image_microphone_retelling.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_microphone_retelling, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_microphone_retelling.started')
            # update status
            image_microphone_retelling.status = STARTED
            image_microphone_retelling.setAutoDraw(True)
        
        # if image_microphone_retelling is active this frame...
        if image_microphone_retelling.status == STARTED:
            # update params
            pass
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in retelling_recordingComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "retelling_recording" ---
    for thisComponent in retelling_recordingComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('retelling_recording.stopped', globalClock.getTime())
    # Run 'End Routine' code from code_9
    if key_resp_retelling.keys != 's':
        recorder.stop_recording()
    # the Routine "retelling_recording" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "reading_instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('reading_instructions.started', globalClock.getTime())
    key_resp_reading_instructions.keys = []
    key_resp_reading_instructions.rt = []
    _key_resp_reading_instructions_allKeys = []
    # keep track of which components have finished
    reading_instructionsComponents = [text_reading_instructions, key_resp_reading_instructions]
    for thisComponent in reading_instructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "reading_instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_reading_instructions* updates
        
        # if text_reading_instructions is starting this frame...
        if text_reading_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_reading_instructions.frameNStart = frameN  # exact frame index
            text_reading_instructions.tStart = t  # local t and not account for scr refresh
            text_reading_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_reading_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_reading_instructions.started')
            # update status
            text_reading_instructions.status = STARTED
            text_reading_instructions.setAutoDraw(True)
        
        # if text_reading_instructions is active this frame...
        if text_reading_instructions.status == STARTED:
            # update params
            pass
        
        # *key_resp_reading_instructions* updates
        waitOnFlip = False
        
        # if key_resp_reading_instructions is starting this frame...
        if key_resp_reading_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_reading_instructions.frameNStart = frameN  # exact frame index
            key_resp_reading_instructions.tStart = t  # local t and not account for scr refresh
            key_resp_reading_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_reading_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_reading_instructions.started')
            # update status
            key_resp_reading_instructions.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_reading_instructions.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_reading_instructions.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_reading_instructions.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_reading_instructions.getKeys(keyList=['space','s'], ignoreKeys=None, waitRelease=False)
            _key_resp_reading_instructions_allKeys.extend(theseKeys)
            if len(_key_resp_reading_instructions_allKeys):
                key_resp_reading_instructions.keys = _key_resp_reading_instructions_allKeys[-1].name  # just the last key pressed
                key_resp_reading_instructions.rt = _key_resp_reading_instructions_allKeys[-1].rt
                key_resp_reading_instructions.duration = _key_resp_reading_instructions_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in reading_instructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "reading_instructions" ---
    for thisComponent in reading_instructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('reading_instructions.stopped', globalClock.getTime())
    # check responses
    if key_resp_reading_instructions.keys in ['', [], None]:  # No response was made
        key_resp_reading_instructions.keys = None
    thisExp.addData('key_resp_reading_instructions.keys',key_resp_reading_instructions.keys)
    if key_resp_reading_instructions.keys != None:  # we had a response
        thisExp.addData('key_resp_reading_instructions.rt', key_resp_reading_instructions.rt)
        thisExp.addData('key_resp_reading_instructions.duration', key_resp_reading_instructions.duration)
    thisExp.nextEntry()
    # the Routine "reading_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "paragraph" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('paragraph.started', globalClock.getTime())
    # skip this Routine if its 'Skip if' condition is True
    continueRoutine = continueRoutine and not (key_resp_reading_instructions.keys == 's')
    # Run 'Begin Routine' code from code_5
    if key_resp_reading_instructions.keys != 's':
        recorder.start_recording(f'data/{expInfo["Nombre"]}/{expInfo["Nombre"]}_LecturaDeParrafo.wav')
    key_resp_reading.keys = []
    key_resp_reading.rt = []
    _key_resp_reading_allKeys = []
    # keep track of which components have finished
    paragraphComponents = [key_resp_reading, paragraph_image]
    for thisComponent in paragraphComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "paragraph" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_resp_reading* updates
        waitOnFlip = False
        
        # if key_resp_reading is starting this frame...
        if key_resp_reading.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_reading.frameNStart = frameN  # exact frame index
            key_resp_reading.tStart = t  # local t and not account for scr refresh
            key_resp_reading.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_reading, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_reading.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_reading.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_reading.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_reading.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_reading.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _key_resp_reading_allKeys.extend(theseKeys)
            if len(_key_resp_reading_allKeys):
                key_resp_reading.keys = _key_resp_reading_allKeys[-1].name  # just the last key pressed
                key_resp_reading.rt = _key_resp_reading_allKeys[-1].rt
                key_resp_reading.duration = _key_resp_reading_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *paragraph_image* updates
        
        # if paragraph_image is starting this frame...
        if paragraph_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            paragraph_image.frameNStart = frameN  # exact frame index
            paragraph_image.tStart = t  # local t and not account for scr refresh
            paragraph_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(paragraph_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'paragraph_image.started')
            # update status
            paragraph_image.status = STARTED
            paragraph_image.setAutoDraw(True)
        
        # if paragraph_image is active this frame...
        if paragraph_image.status == STARTED:
            # update params
            pass
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in paragraphComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "paragraph" ---
    for thisComponent in paragraphComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('paragraph.stopped', globalClock.getTime())
    # Run 'End Routine' code from code_5
    if key_resp_reading_instructions.keys != 's':
        recorder.stop_recording()
    # the Routine "paragraph" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "letter_A_instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('letter_A_instructions.started', globalClock.getTime())
    key_resp_letter_A_instructions.keys = []
    key_resp_letter_A_instructions.rt = []
    _key_resp_letter_A_instructions_allKeys = []
    # keep track of which components have finished
    letter_A_instructionsComponents = [text_letter_A_instructions, key_resp_letter_A_instructions]
    for thisComponent in letter_A_instructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "letter_A_instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_letter_A_instructions* updates
        
        # if text_letter_A_instructions is starting this frame...
        if text_letter_A_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_letter_A_instructions.frameNStart = frameN  # exact frame index
            text_letter_A_instructions.tStart = t  # local t and not account for scr refresh
            text_letter_A_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_letter_A_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_letter_A_instructions.started')
            # update status
            text_letter_A_instructions.status = STARTED
            text_letter_A_instructions.setAutoDraw(True)
        
        # if text_letter_A_instructions is active this frame...
        if text_letter_A_instructions.status == STARTED:
            # update params
            pass
        
        # *key_resp_letter_A_instructions* updates
        waitOnFlip = False
        
        # if key_resp_letter_A_instructions is starting this frame...
        if key_resp_letter_A_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_letter_A_instructions.frameNStart = frameN  # exact frame index
            key_resp_letter_A_instructions.tStart = t  # local t and not account for scr refresh
            key_resp_letter_A_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_letter_A_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_letter_A_instructions.started')
            # update status
            key_resp_letter_A_instructions.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_letter_A_instructions.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_letter_A_instructions.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_letter_A_instructions.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_letter_A_instructions.getKeys(keyList=['space','s'], ignoreKeys=None, waitRelease=False)
            _key_resp_letter_A_instructions_allKeys.extend(theseKeys)
            if len(_key_resp_letter_A_instructions_allKeys):
                key_resp_letter_A_instructions.keys = _key_resp_letter_A_instructions_allKeys[-1].name  # just the last key pressed
                key_resp_letter_A_instructions.rt = _key_resp_letter_A_instructions_allKeys[-1].rt
                key_resp_letter_A_instructions.duration = _key_resp_letter_A_instructions_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in letter_A_instructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "letter_A_instructions" ---
    for thisComponent in letter_A_instructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('letter_A_instructions.stopped', globalClock.getTime())
    # check responses
    if key_resp_letter_A_instructions.keys in ['', [], None]:  # No response was made
        key_resp_letter_A_instructions.keys = None
    thisExp.addData('key_resp_letter_A_instructions.keys',key_resp_letter_A_instructions.keys)
    if key_resp_letter_A_instructions.keys != None:  # we had a response
        thisExp.addData('key_resp_letter_A_instructions.rt', key_resp_letter_A_instructions.rt)
        thisExp.addData('key_resp_letter_A_instructions.duration', key_resp_letter_A_instructions.duration)
    thisExp.nextEntry()
    # the Routine "letter_A_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "letter_A_recording" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('letter_A_recording.started', globalClock.getTime())
    # skip this Routine if its 'Skip if' condition is True
    continueRoutine = continueRoutine and not (key_resp_letter_A_instructions.keys == 's')
    key_resp_letter_A.keys = []
    key_resp_letter_A.rt = []
    _key_resp_letter_A_allKeys = []
    # Run 'Begin Routine' code from code_10
    if key_resp_letter_A_instructions.keys != 's':
        recorder.start_recording(f'data/{expInfo["Nombre"]}/{expInfo["Nombre"]}_VocalA.wav')
    # keep track of which components have finished
    letter_A_recordingComponents = [key_resp_letter_A, image_microphone_letter_A]
    for thisComponent in letter_A_recordingComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "letter_A_recording" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_resp_letter_A* updates
        waitOnFlip = False
        
        # if key_resp_letter_A is starting this frame...
        if key_resp_letter_A.status == NOT_STARTED and tThisFlip >= 5.5-frameTolerance:
            # keep track of start time/frame for later
            key_resp_letter_A.frameNStart = frameN  # exact frame index
            key_resp_letter_A.tStart = t  # local t and not account for scr refresh
            key_resp_letter_A.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_letter_A, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_letter_A.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_letter_A.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_letter_A.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_letter_A.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_letter_A.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _key_resp_letter_A_allKeys.extend(theseKeys)
            if len(_key_resp_letter_A_allKeys):
                key_resp_letter_A.keys = _key_resp_letter_A_allKeys[-1].name  # just the last key pressed
                key_resp_letter_A.rt = _key_resp_letter_A_allKeys[-1].rt
                key_resp_letter_A.duration = _key_resp_letter_A_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *image_microphone_letter_A* updates
        
        # if image_microphone_letter_A is starting this frame...
        if image_microphone_letter_A.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            image_microphone_letter_A.frameNStart = frameN  # exact frame index
            image_microphone_letter_A.tStart = t  # local t and not account for scr refresh
            image_microphone_letter_A.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_microphone_letter_A, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_microphone_letter_A.started')
            # update status
            image_microphone_letter_A.status = STARTED
            image_microphone_letter_A.setAutoDraw(True)
        
        # if image_microphone_letter_A is active this frame...
        if image_microphone_letter_A.status == STARTED:
            # update params
            pass
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in letter_A_recordingComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "letter_A_recording" ---
    for thisComponent in letter_A_recordingComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('letter_A_recording.stopped', globalClock.getTime())
    # Run 'End Routine' code from code_10
    if key_resp_letter_A_instructions.keys != 's':
        recorder.stop_recording()
    # the Routine "letter_A_recording" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "pataka_instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('pataka_instructions.started', globalClock.getTime())
    key_resp_pataka_instructions.keys = []
    key_resp_pataka_instructions.rt = []
    _key_resp_pataka_instructions_allKeys = []
    # keep track of which components have finished
    pataka_instructionsComponents = [text_pataka_instructions, key_resp_pataka_instructions]
    for thisComponent in pataka_instructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "pataka_instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_pataka_instructions* updates
        
        # if text_pataka_instructions is starting this frame...
        if text_pataka_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_pataka_instructions.frameNStart = frameN  # exact frame index
            text_pataka_instructions.tStart = t  # local t and not account for scr refresh
            text_pataka_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_pataka_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_pataka_instructions.started')
            # update status
            text_pataka_instructions.status = STARTED
            text_pataka_instructions.setAutoDraw(True)
        
        # if text_pataka_instructions is active this frame...
        if text_pataka_instructions.status == STARTED:
            # update params
            pass
        
        # *key_resp_pataka_instructions* updates
        waitOnFlip = False
        
        # if key_resp_pataka_instructions is starting this frame...
        if key_resp_pataka_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_pataka_instructions.frameNStart = frameN  # exact frame index
            key_resp_pataka_instructions.tStart = t  # local t and not account for scr refresh
            key_resp_pataka_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_pataka_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_pataka_instructions.started')
            # update status
            key_resp_pataka_instructions.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_pataka_instructions.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_pataka_instructions.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_pataka_instructions.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_pataka_instructions.getKeys(keyList=['space','s'], ignoreKeys=None, waitRelease=False)
            _key_resp_pataka_instructions_allKeys.extend(theseKeys)
            if len(_key_resp_pataka_instructions_allKeys):
                key_resp_pataka_instructions.keys = _key_resp_pataka_instructions_allKeys[-1].name  # just the last key pressed
                key_resp_pataka_instructions.rt = _key_resp_pataka_instructions_allKeys[-1].rt
                key_resp_pataka_instructions.duration = _key_resp_pataka_instructions_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in pataka_instructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "pataka_instructions" ---
    for thisComponent in pataka_instructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('pataka_instructions.stopped', globalClock.getTime())
    # check responses
    if key_resp_pataka_instructions.keys in ['', [], None]:  # No response was made
        key_resp_pataka_instructions.keys = None
    thisExp.addData('key_resp_pataka_instructions.keys',key_resp_pataka_instructions.keys)
    if key_resp_pataka_instructions.keys != None:  # we had a response
        thisExp.addData('key_resp_pataka_instructions.rt', key_resp_pataka_instructions.rt)
        thisExp.addData('key_resp_pataka_instructions.duration', key_resp_pataka_instructions.duration)
    thisExp.nextEntry()
    # the Routine "pataka_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "pataka_recording" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('pataka_recording.started', globalClock.getTime())
    # skip this Routine if its 'Skip if' condition is True
    continueRoutine = continueRoutine and not (key_resp_pataka_instructions.keys == 's')
    key_resp_pataka.keys = []
    key_resp_pataka.rt = []
    _key_resp_pataka_allKeys = []
    # Run 'Begin Routine' code from code_11
    if key_resp_pataka_instructions.keys != 's':
        recorder.start_recording(f'data/{expInfo["Nombre"]}/{expInfo["Nombre"]}_Pataka.wav')
    # keep track of which components have finished
    pataka_recordingComponents = [key_resp_pataka, image_microphone_pataka]
    for thisComponent in pataka_recordingComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "pataka_recording" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_resp_pataka* updates
        waitOnFlip = False
        
        # if key_resp_pataka is starting this frame...
        if key_resp_pataka.status == NOT_STARTED and tThisFlip >= 5.5-frameTolerance:
            # keep track of start time/frame for later
            key_resp_pataka.frameNStart = frameN  # exact frame index
            key_resp_pataka.tStart = t  # local t and not account for scr refresh
            key_resp_pataka.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_pataka, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_pataka.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_pataka.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_pataka.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_pataka.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_pataka.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _key_resp_pataka_allKeys.extend(theseKeys)
            if len(_key_resp_pataka_allKeys):
                key_resp_pataka.keys = _key_resp_pataka_allKeys[-1].name  # just the last key pressed
                key_resp_pataka.rt = _key_resp_pataka_allKeys[-1].rt
                key_resp_pataka.duration = _key_resp_pataka_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *image_microphone_pataka* updates
        
        # if image_microphone_pataka is starting this frame...
        if image_microphone_pataka.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            image_microphone_pataka.frameNStart = frameN  # exact frame index
            image_microphone_pataka.tStart = t  # local t and not account for scr refresh
            image_microphone_pataka.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_microphone_pataka, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_microphone_pataka.started')
            # update status
            image_microphone_pataka.status = STARTED
            image_microphone_pataka.setAutoDraw(True)
        
        # if image_microphone_pataka is active this frame...
        if image_microphone_pataka.status == STARTED:
            # update params
            pass
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in pataka_recordingComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "pataka_recording" ---
    for thisComponent in pataka_recordingComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('pataka_recording.stopped', globalClock.getTime())
    # Run 'End Routine' code from code_11
    if key_resp_pataka_instructions.keys != 's':
        recorder.stop_recording()
    # the Routine "pataka_recording" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "acknowledgment" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('acknowledgment.started', globalClock.getTime())
    key_resp_end.keys = []
    key_resp_end.rt = []
    _key_resp_end_allKeys = []
    # keep track of which components have finished
    acknowledgmentComponents = [text_end, key_resp_end]
    for thisComponent in acknowledgmentComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "acknowledgment" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_end* updates
        
        # if text_end is starting this frame...
        if text_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_end.frameNStart = frameN  # exact frame index
            text_end.tStart = t  # local t and not account for scr refresh
            text_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_end, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_end.started')
            # update status
            text_end.status = STARTED
            text_end.setAutoDraw(True)
        
        # if text_end is active this frame...
        if text_end.status == STARTED:
            # update params
            pass
        
        # *key_resp_end* updates
        waitOnFlip = False
        
        # if key_resp_end is starting this frame...
        if key_resp_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_end.frameNStart = frameN  # exact frame index
            key_resp_end.tStart = t  # local t and not account for scr refresh
            key_resp_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_end, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_end.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_end.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_end.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_end.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_end.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _key_resp_end_allKeys.extend(theseKeys)
            if len(_key_resp_end_allKeys):
                key_resp_end.keys = _key_resp_end_allKeys[-1].name  # just the last key pressed
                key_resp_end.rt = _key_resp_end_allKeys[-1].rt
                key_resp_end.duration = _key_resp_end_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in acknowledgmentComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "acknowledgment" ---
    for thisComponent in acknowledgmentComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('acknowledgment.stopped', globalClock.getTime())
    # the Routine "acknowledgment" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    # Run 'End Experiment' code from code_general
    recorder._serial.close()
    
    tempFileName = thisExp.dataFileName + '_TEMP.csv' 
    thisExp.saveAsWideText(tempFileName)    
    df = pd.read_csv(tempFileName, delimiter=',')
    
    df = df.fillna(method='bfill')
    df = df.fillna(method='ffill')
    idx = len(df) * [False]
    idx[0] = True
    
    # Select the first row
    first_row_df = df[idx]
    
    first_row_df = first_row_df.dropna(axis=1, how='all')
    first_row_df.to_csv(thisExp.dataFileName + '_spanish.csv', sep=';', decimal=',', index=False)
    
    os.remove(tempFileName)
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
