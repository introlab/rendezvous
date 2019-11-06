## Prerequisites

- Ubuntu 18.04.3 LTS (Bionic Beaver)

## 1 - Update Ubuntu

Make sure ubuntu is up to date

    sudo apt-get update
    sudo apt-get upgrade
    sudo apt autoremove

## 1 - Qt

Download the latest version of qt online installer: https://download.qt.io/archive/online_installers/3.1/

Make the installer runnable

    chmod +x qt-unified-linux-x64-3.1.1-online.run

Run the installer

    ./qt-unified-linux-x64-3.1.1-online.run

Select `Qt 5.12.5` then launch the installation

## 2 - Environnement

Create working directories

    cd ~
    mkdir dev
    mkdir dev/workspace
    mkdir dev/lib
