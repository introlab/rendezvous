![LOGO](https://github.com/introlab/rendezvous/blob/master/screenshots/logo.svg)

[![Build Status](https://travis-ci.org/introlab/rendezvous.svg?branch=master)](https://travis-ci.org/introlab/rendezvous)

## Introduction

For our end of degree project in computer engineering at the University of Sherbrooke, the RendezVoUS team is happy to present you the next generation video-conference system : Steno!

## Description

No more video-conferences filled with technical problems and meetings with multiple people behind the same laptop! 

Steno is an intelligent video-conferencing system that allows you to make worry-free calls by using 16 microphones and a 360° camera. The system use artificial intelligence as well as other technologies to offer an enhanced experience both visually and audibly.

Steno film and record each person individually, then sends everything to the video-conferencing system. It's like if everyone had an webcam! Steno films you dynamically, which means that if you move around the table, it will be able track your voice and position, and make sure all your words and mouvements are sent to other end of the video-conference.

## Specifications

### Hardware

- Platform : [Jetson TX2](https://developer.nvidia.com/embedded/jetson-tx2-developer-kit)

- Sound card : [16SoundsUSB](https://github.com/introlab/16SoundsUSB)

- Microphones (x16) : [xSoundsMicrophones](https://github.com/introlab/xSoundsMicrophones)

- Audio amplifiers (x2) : [Stereo 20W Class D Audio Amplifier - MAX9744](https://www.adafruit.com/product/1752)

- Speakers (x4) : [Yeeco 2.5 inch 4 ohm stero audio speakers](https://www.amazon.ca/dp/B075B72J5F/ref=pe_3034960_233709270_TE_item)

- Camera : [See3CAM_CU135 – 4K Custom Lens USB 3.0 Camera Board (Color)](https://www.e-consystems.com/4k-usb-camera.asp)

- Lens : [Cvivid Lenses 1.2mm Wide Angle 220 Degree Fisheye](https://www.amazon.ca/dp/B07DN9542G/ref=pe_3034960_233709270_TE_item)

### Software

- Operating system : [JetPack 4.2.2](https://developer.nvidia.com/embedded/jetpack-archive)

- Sound source localisation, tracking, separation and post-filtering : [ODAS](https://github.com/introlab/odas)

- Neural network framework : [Darknet](https://github.com/pjreddie/darknet) 

- Real-time person detection : [YOLO](https://github.com/pjreddie/darknet/wiki/YOLO:-Real-Time-Object-Detection)


- Echo and noise cancelling : [Pulse Audio](https://www.freedesktop.org/wiki/Software/PulseAudio/)

- Virtual devices : [V4l2](https://github.com/mpromonet/libv4l2cpp)

- WebRTC video-conference web server : [Jitsi Meet](https://github.com/jitsi/jitsi-meet)

- Graphhical user interfaces : [Qt](https://www.qt.io/)

## The Prototype

![](https://github.com/introlab/rendezvous/blob/master/screenshots/montage.jpg)

## Build and Run

See the [wiki](https://github.com/introlab/rendezvous/wiki) for more.

## Licensing

GNU General Public License v3.0; see [`LICENSE`](LICENSE) for details.