# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:42:04 2021

@author: OBRIEJ25
"""
import numpy as np
import sys

#%% Get guitar chord data
# Data from 'https://gist.github.com/ftes/cccf63d5d4cd61b60f594835755c177c#file-guitar-chords-csv'
data = np.genfromtxt('./guitar_tab_info.csv', dtype='str', comments=None,
                     delimiter=',', autostrip=True)

# We dont need to calculate for higher frets for this chord list
max_fret = 0  # = 9
for row in data[1:,1:].flatten():
    test = max(np.array(list(row.replace('X','').replace(':','').replace(')','').replace('(',''))).astype(int))
    if test > max_fret: max_fret = test

datum, types = data[1:, :], data[0,1:]

# Create a dictionary of dictionaries, specifying chords and the sub-type
d = {}
for row in datum:
    chord = row[0]
    d[chord] = {}
    for c_type,info in zip(types,row[1:]):
        d[chord][c_type] = info

num_c, num_t = len(datum[:,0]), len(types)

# Number of dots on each fret position on guitar neck
fret_dots = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 2, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 2]



#%% Synthesize the waveforms for each of the string/fret combos that we need
from karplus_strong_guitar_sound_generator import karplus_strong_jit

from scipy.fftpack import fft, ifft, fftfreq
fret = np.arange(max_fret+1)
freq = np.array([82.41, 87.31, 92.50, 98.00, 103.83, 110.00, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94, 110.00, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 196.00, 207.65, 220.00, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, 523.25, 554.37, 587.33, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.26, 698.46, 739.99, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.26, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77]).reshape(6,20).transpose()
freq = freq[:max_fret, :]

fs = 44100
wavetable_size = (fs / freq).astype(int)

# Low and High pass filter frequencies
low_filter, high_filter = 5, 8000

# Stretch factors, lets try a mapping from 1 (for lowest freq) to 2 (for highest freq)
def my_map(x, lst): return (x - x.min()) * (lst[1] - lst[0]) / (x.max() - x.min()) + lst[0]
stretch_factors = my_map(freq, [1,4])

# Fade to use: Logarithmic, 1/x, linear (same as 1/x with base=1)
log_fade = my_map( np.log(2*fs - np.arange(2*fs)) + np.log(2*fs - np.arange(2*fs))[::-1], [0,1])
sin_fade_len = 1000
sin_ramp = 0.5*(np.sin(np.linspace(3*np.pi/2, np.pi/2, sin_fade_len)) + 1)
sin_fade = np.hstack(( sin_ramp, np.ones(2*fs - 2*len(sin_ramp)), sin_ramp[::-1] ))

# Matrices for all the waveforms (floats) and FFTs (complex numbers)
samples, samples_filtered = [np.empty((*freq.shape,2*fs)) for i in [0,0]]
ffts, ffts_filtered = [np.empty((*freq.shape,2*fs), dtype=np.complex128) for i in [0,0]]

x_freq = fftfreq(2*fs, 1/fs)  # Get frequency units to apply the high/low pass filters later


pure_sine = 0
if not pure_sine:
    for i in range(freq.shape[0]):
        for j in range(freq.shape[1]):
            # A random distribution of -1 and 1 turns out to give a string sound.
            wavetable = (2 * np.random.randint(0, 2, wavetable_size[i,j]) - 1).astype(float)
            # Sine/cosine wavetable kind sounds like a distorted electric guitar
            #wavetable = np.cos(my_map( np.arange(wavetable_size[i,j]), [0, 2*np.pi]))

            samples[i,j] = karplus_strong_jit(wavetable, 2*fs, stretch_factors[i,j])
            ffts[i,j] = fft(samples[i,j])

            ffts_filtered[i,j] = 1 * ffts[i,j]
            ffts_filtered[i,j][(abs(x_freq)<low_filter) | (abs(x_freq)>high_filter)] = 0
            samples_filtered[i,j] = np.real(ifft(ffts_filtered[i,j]))

else: #Use a pure sine wave instead
    for i in range(freq.shape[0]):
        for j in range(freq.shape[1]):
            samples_filtered[i,j] = np.sin( 2*np.pi * freq[i,j] * np.linspace(0, 2, 2*fs) )

samples_filtered_log_fade = samples_filtered * log_fade
samples_filtered_sin_fade = samples_filtered * sin_fade

#final_samples = samples_filtered_log_fade / np.max(samples_filtered_log_fade, axis=-1)[:,:,None]
final_samples = samples_filtered_sin_fade / np.max(samples_filtered_sin_fade, axis=-1)[:,:,None]


#%% Make the pygame applicaiton
import pygame as pg
pg.init()
pg.mixer.init()
WIN_SIZE = (1000, 700)
BCKGD, BLACK, WHITE = (230,242,255), (0,0,0), (255,255,255)
RED, GREEN, BLUE, YELLOW = (255,0,0), (0,255,0), (0,0,255), (255,255,0)
GRN, RD = (100,150,150), (200,0,0)


def fade(tup, opac=100):
    tup = (*tup, opac)
    return tup


# Functions to draw chord shape
def draw_chord(win, chord_info):
    # Initial formatting of chord info and drawing fret number
    pos = chord_info.split(':')
    fret = 0 if len(pos[0]) > 1 else int(pos[0])
    if len(pos[0]) == 1:
        txt = f'{pos[0]}th fret'
        text = pg.font.SysFont('Times New Roman', 40).render(txt, 1, GRN)
        move_right = 100
        win.blit(text, (720 +move_right -round(text.get_width()/2), 110 -round(text.get_height()/2)) )

    # Need to include logic to draw a barre:
    # Check if brackets are present
    # Get first and last numbers inside the brackets
    # Draw the first and last circles and a rectangle in between connecting them
    # Draw the finger number on the left (first circle) or in rectangle centre

    # Also add code to plot fret markers
    for j,(i,frt), in enumerate(zip(fret_dots[fret:fret+5], np.arange(fret+1, fret+6))):
        if i == 0: continue
        elif i == 1: x, y = [495], 110*(j+1.5)
        elif i == 2: x, y = [315, 675], 110*(j+0.5)
        for x0 in x:
            pg.draw.circle(win, fade(YELLOW, 100), (x0,y), 20, width=0)
            text = pg.font.SysFont('Times New Roman', 36).render(str(frt), 1, fade(GRN, 100))
            win.blit(text, (x0 - round(text.get_width()/2), y - round(text.get_height()/2)) )

    # Fret positions
    labs = pos[0] if len(pos[0])>1 else pos[1]
    nums = np.array(list(labs.replace('X','0').replace('(','').replace(')',''))).astype(int)
    x_pos, y_pos = np.arange(270, 271+5*90, 90), 110 + 110*nums - 55
    labs = list(labs)

    # Whether finger number is to be included
    if len(chord_info) > 12:
        fings = np.array(list(pos[-1])).astype(int)
    else: fings = np.zeros(6)

    for num,lab,x,y,fing in zip(nums,labs,x_pos,y_pos,fings):
        # Black circle if a fret is given
        if num >= 1: pg.draw.circle(win, BLACK, (x,y), 20, width=0)
        # Open circle is fret number is 0
        elif lab == '0': pg.draw.circle(win, BLACK, (x,y+30), 20, width=3)
        else:   # X is fret is not played
            text = pg.font.SysFont('Times New Roman', 40).render(lab, 1, BLACK)
            win.blit(text, (x - round(text.get_width()/2), y - round(text.get_height()/2)+32) )
        if fing:   # Draw fingering for non-zero fingers/frets
            text = pg.font.SysFont('Times New Roman', 36).render(str(fing), 1, WHITE)
            win.blit(text, (x - round(text.get_width()/2), y - round(text.get_height()/2)) )


# Function to play chord using the synthesised waveform
def play_chord(chord_info, pick_delay=20):
    # Initial formatting of chord info and drawing fret number
    pos = chord_info.split(':')
    frets = pos[0] if len(pos[0]) > 1 else pos[1]
    frets = frets.replace('(','').replace(')','')

    strngs, frts = [], []
    for i in range(6):
        if frets[i] == 'X': continue
        strngs.append(i), frts.append(int(frets[i]))

    waveforms = final_samples[frts, strngs]

    #"""
    # To implement the delay, check how much to pad for each note, from pick_delay and fs
    sample_pad = int(pick_delay * 1e-3 * 1 * fs)

    waveform = np.zeros(2*fs + (len(frts)-1)*sample_pad)
    for i in range(len(frts)):
        wf = np.hstack(( np.zeros(i*sample_pad), waveforms[i], np.zeros((len(frts)-1-i)*sample_pad) ))
        waveform +=wf
    waveform /= len(frts)

    sound = np.asarray([32767*waveform, 32767*waveform]).T.astype(np.int16)
    sound = pg.sndarray.make_sound(sound.copy()) # .copy() makes the array contiguous in memory
    # play the sound with optional delay to mimic strumming/picking
    sound.set_volume(0.33)
    sound.play()

    """

    for waveform in waveforms:
        # Convert the sound wave to a pygame sound object
        sound = np.asarray([32767*waveform, 32767*waveform]).T.astype(np.int16) # no idea why the *32767 or the int16
        sound = pg.sndarray.make_sound(sound.copy()) # .copy() makes the array contiguous in memory
        # play the sound with optional delay to mimic strumming/picking
        sound.set_volume(0.50 / len(waveforms))
        sound.play()
        pg.time.wait(pick_delay)

    #"""


# Define the button class
class button():
    def __init__(self, text, x, y, w=150, h=100, bg_clr=BLACK, txt_clr=WHITE):
        self.text = text
        self.x = x
        self.y = y
        self.bg_clr = bg_clr
        self.txt_clr = txt_clr
        self.width = w
        self.height = h
        self.rect = pg.Rect(self.x, self.y, self.width, self.height)

    def draw(self, win, f_sz=34):
        pg.draw.rect(win, self.bg_clr, self.rect)
        font = pg.font.SysFont('Times New Roman', f_sz)
        text = font.render(self.text, 1, self.txt_clr)
        # Some maths to center the text in the button, using the size of the button and text
        win.blit(text, ( self.x + round(self.width/2) - round(text.get_width()/2) ,\
                        self.y + round(self.height/2) - round(text.get_height()/2) ))

    def clicked(self, pos):
            if self.rect.collidepoint(pos):  # check if click position is inside the button
                return True
            else: return False


class line():
    def __init__(self, ori, x0, y0, length, thickness):
        self.x0, self.y0 = x0, y0
        if ori == 'string': self.clr = RD
        elif ori == 'fret': self.clr = GRN
        self.rect = pg.Rect(self.x0, self.y0, length, thickness)

    def draw(self, win):
        pg.draw.rect(win, self.clr, self.rect)


def main():
    window = pg.display.set_mode(WIN_SIZE)
    clock = pg.time.Clock()

    # Set up the button positions
    chord_btns, type_btns = [], []
    for i,txt in enumerate(datum[:,0]):
        ydiv = WIN_SIZE[1] / ( num_c + 3 + (num_c-1)/4 )
        xdiv = 60
        btn = button(txt, ydiv, (2+(i*1.25))*ydiv, xdiv, ydiv)
        chord_btns.append(btn)

    for i,txt in enumerate(types):
        xdiv = WIN_SIZE[0] / ( num_t + 2 + (num_t-1)/5 )
        btn = button(txt, (i*1.20)*xdiv+95, 10, xdiv, ydiv)
        type_btns.append(btn)

    play_chord_btn = button("Play Chord", 770, 295, 180, 50)
    play_notes_btn = button("Play Notes", 770, 355, 180, 50)


    # Set up the strings and frets and dots pictures
    strings, frets = [], []
    for i in range(6):
        x, y = 90*(i+3), 110
        ln = line('string', x,y, 2, 550+4)
        strings.append(ln)
    for i in range(6):
        x, y = 270, 110*(i+1)
        thick = 10 if i==0 else 4
        ln = line('fret', x,y, 452, thick)
        frets.append(ln)
    lines = frets + strings

    # Setup circle positions
    run = True
    chord, c_type = None, None
    while run:
        for event in pg.event.get():

            if event.type == pg.QUIT:
                run = False
                pg.quit()
                sys.exit()

            if event.type == pg.MOUSEBUTTONDOWN:
                # Loop through chord buttons (on left)
                for btn in chord_btns:
                    if btn.clicked(event.pos):  # Check which, if any, chord btn clicked -> store
                        clicked_btn, chord = btn, btn.text
                        btn.bg_clr, btn.txt_clr = YELLOW, BLACK
                        for btn in chord_btns:      # Set all not currently active btns to default clr
                            if btn == clicked_btn: continue
                            btn.bg_clr, btn.txt_clr = BLACK, WHITE
                # Loop through type buttons (scross top)
                for btn in type_btns:
                    if btn.clicked(event.pos):
                        clicked_type, c_type = btn, btn.text
                        btn.bg_clr, btn.txt_clr = YELLOW, BLACK
                        for btn in type_btns:
                            if btn == clicked_type: continue
                            btn.bg_clr, btn.txt_clr = BLACK, WHITE
                # Check play sound buttons
                for btn in [play_chord_btn, play_notes_btn]:
                    if btn.clicked(event.pos):
                        if "Chord" in btn.text:
                            if chord and c_type: play_chord(d[chord][c_type], 20)
                        elif "Notes" in btn.text:
                            if chord and c_type: play_chord(d[chord][c_type], 350)
                        btn.bg_clr, btn.txt_clr = YELLOW, BLACK
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Need to get the button to just flash yellow or something when pressed, then revert
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Make sure the draw order is correct, window -> frets -> strings -> tabs etc.
        window.fill(BCKGD)

        for btn in chord_btns: btn.draw(window)
        for btn in type_btns: btn.draw(window, f_sz=30)
        for btn in [play_chord_btn, play_notes_btn]: btn.draw(window)

        for ln in lines: ln.draw(window)

        if chord and c_type:
            draw_chord(window, d[chord][c_type])
            if not pg.mixer.get_busy(): # Turn the play buttons back to default colour
                for btn in [play_chord_btn, play_notes_btn]: btn.bg_clr, btn.txt_clr = BLACK, WHITE

        pg.display.flip()  # update the entire display surface to the screen

        clock.tick(30)


#%%
if __name__ == '__main__':
    main()
    pg.quit()


