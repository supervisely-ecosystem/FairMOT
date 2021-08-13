<div align="center" markdown>

<img src="https://imgur.com/GbTR1UA.png"/>  

# Serve FairMOT

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Use">How To Use</a> •
  <a href="#Watch-Demo-Video">Demo</a> •
    <a href="#Screenshots">Screenshots</a>
</p>

[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/FairMOT)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/FairMOT&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/FairMOT&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/FairMOT&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

Serve FairMOT model as REST API.

Application key points:
- Can be deployed on CPU or GPU
- Can be used in Supervisely APPs
- Available via REST API



# How to Use

1. Prepare a video project containing objects annotated with `Rectangle shapes`.  
You can use the [import MOT format APP](https://ecosystem.supervise.ly/apps/import-mot-format) to import your [MOT format data](https://motchallenge.net/instructions/) into Supervisely.  

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/import-mot-format" src="https://imgur.com/gYUrNc2.png" width="350px" style='padding-bottom: 10px'/>


2. Add app from ecosystem to your team  
<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/FairMOT/supervisely/train" src="https://imgur.com/Mk1gpGJ.png" width="350px" style='padding-bottom: 10px'/>


3. Run app from the context menu of **video project** with labeled objects:  
<img src="https://imgur.com/o7ZJxpd.png" width="80%" style='padding-top: 10px'>  


4. Set the settings and start the train



# Watch Demo Video

`in developing`
<!--
<a data-key="sly-embeded-video-link" href="https://youtu.be/yvWegId-edU" data-video-code="yvWegId-edU">
    <img src="https://imgur.com/VRQdPXx.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;">
</a> -->


# Screenshots
<img src="https://imgur.com/DX1Wym9.png" width="auto" style='padding-top: 10px'>
