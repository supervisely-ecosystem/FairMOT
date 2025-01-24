<div align="center" markdown>

<img src="https://imgur.com/GbTR1UA.png"/>  

# Serve FairMOT

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Use">How To Use</a> •
  <a href="#Watch-Demo-Video">Demo</a> •
    <a href="#For-Developers">For Developers</a>
</p>

[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/FairMOT)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/fairmot/supervisely/serve.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/fairmot/supervisely/serve.png)](https://supervisely.com)

</div>

# Overview

Serve FairMOT model as Supervisely APP.

Application key points:
- Can be deployed on CPU or GPU
- Can be used in Supervisely APPs
- Available via REST API



# How to Use

1. Train your model using [Train FairMOT APP](https://ecosystem.supervisely.com/apps/supervisely-ecosystem%252Ffairmot%252Fsupervisely%252Ftrain).  
After the end of the train, you get a folder with checkpoints available in [Supervisely Files](https://app.supervisely.com/files/) by `/FairMOT/train/{experiment_name}/checkpoints` path.

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/FairMOT/supervisely/train" src="https://imgur.com/Mk1gpGJ.png" width="350px" style='padding-bottom: 10px'/>


2. Add [Serve FairMOT](https://ecosystem.supervisely.com/apps/supervisely-ecosystem%252Ffairmot%252Fsupervisely%252Fserve) from ecosystem to your team  

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/FairMOT/supervisely/serve" src="https://imgur.com/ksDJmF0.png" width="350px" style='padding-bottom: 10px'/>

3. Open checkpoints dir `/FairMOT/train/{experiment_name}/checkpoints` in [Supervisely Files](https://app.supervisely.com/files/) and **Run Application**.  
<img src="https://imgur.com/fKOoOvg.png" width="80%" style='padding-top: 10px'>  

4. The model has been successfully deployed  
<img src="https://imgur.com/1v8EYKR.png" width="80%" style='padding-top: 10px'>  



# Watch Demo Video

`in developing`
<!--
<a data-key="sly-embeded-video-link" href="https://youtu.be/yvWegId-edU" data-video-code="yvWegId-edU">
    <img src="https://imgur.com/VRQdPXx.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;">
</a> -->


# For Developers



[Check this Python Example](https://github.com/supervisely-ecosystem/FairMOT/blob/master/supervisely/serve/src/demo_api_requests.py). It illustrates available methods of the deployed model. Now you can integrate network predictions to your python script. This is the way how other Supervisely Apps can communicate with NNs. And also you can use serving app as an example — how to use downloaded NN weights outside Supervisely.
