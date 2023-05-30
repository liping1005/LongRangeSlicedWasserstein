# Long Range Constraints for Neural Texture Synthesis Using Sliced Wasserstein Loss
This is the project page for "Long Range Constraints for Neural Texture Synthesis Using Sliced Wasserstein Loss." We propose an improvement of "A Sliced Wasserstein Loss for Neural Texture Synthesis" by Heitz et al. by adding an additional loss term to capture nonstationary statistics in textures. Here are some examples of our synthesis with comparisions other algorithms. **Left:** Reference, **Second Column:** Original SW Synthesis, **Third Column:** Spectrum Constraint, **Right:** Our Algorithm (K=0).
<p align=center>
<img src="refs/img_2.jpg"  width="200" height="200">
<img src="heitz/result_2.jpg" width="200" height="200">
<img src="spectrum/result_2.jpg"  width="200" height="200">
 <img src="scale_0/result_2.jpg"  width="200" height="200">

<img src="refs/img_7.jpg"  width="200" height="200">
<img src="heitz/result_7.jpg" width="200" height="200">
<img src="spectrum/result_7.jpg"  width="200" height="200">
<img src="scale_0/result_7.jpg"  width="200" height="200">
  
<img src="refs/img_9.jpg"  width="200" height="200">
<img src="heitz/result_9.jpg" width="200" height="200">
<img src="spectrum/result_9.jpg"  width="200" height="200">
<img src="scale_0/result_9.jpg"  width="200" height="200">

<img src="refs/img_11.jpg"  width="200" height="200">
<img src="heitz/result_11.jpg" width="200" height="200">
<img src="spectrum/result_11.jpg"  width="200" height="200">
<img src="scale_0/result_11.jpg"  width="200" height="200">
  
<img src="refs/img_16.jpg"  width="200" height="200">
<img src="heitz/result_16.jpg" width="200" height="200">
<img src="spectrum/result_16.jpg"  width="200" height="200">
<img src="scale_0/result_16.jpg"  width="200" height="200">
  
<img src="refs/img_20.jpg"  width="200" height="200">
<img src="heitz/result_20.jpg" width="200" height="200">
<img src="spectrum/result_20.jpg"  width="200" height="200">
<img src="scale_0/result_20.jpg"  width="200" height="200">
  
 <img src="refs/img_21.jpg"  width="200" height="200">
<img src="heitz/result_21.jpg" width="200" height="200">
<img src="spectrum/result_21.jpg"  width="200" height="200">
<img src="scale_0/result_21.jpg"  width="200" height="200">
  
<img src="refs/img_23.jpg"  width="200" height="200">
<img src="heitz/result_23.jpg" width="200" height="200">
<img src="spectrum/result_23.jpg"  width="200" height="200">
<img src="scale_0/result_23.jpg"  width="200" height="200">

<img src="refs/img_24.jpg"  width="200" height="200">
<img src="heitz/result_24.jpg" width="200" height="200">
<img src="spectrum/result_24.jpg"  width="200" height="200">
<img src="scale_0/result_24.jpg"  width="200" height="200">
  
 <img src="refs/img_31.jpg"  width="200" height="200">
<img src="heitz/result_31.jpg" width="200" height="200">
<img src="spectrum/result_31.jpg"  width="200" height="200">
<img src="scale_0/result_31.jpg"  width="200" height="200">
  
<img src="refs/img_34.jpg"  width="200" height="200">
<img src="heitz/result_34.jpg" width="200" height="200">
<img src="spectrum/result_34.jpg"  width="200" height="200">
 <img src="scale_0/result_34.jpg"  width="200" height="200">
 

The texures above were mostly periodic or pseudoperiodic, so let's test our algorithm on some textures have diffent long range constraints/less stationary features. For our comparison, we compare our alogorithm with K = 1 against Heitz et. al. and with Gonthier et. al. **Left:** Reference, **Second Column:** Original SW Synthesis, **Third Column:** Gonthier, **Right:** Our Algorithm (K=1).

<p align=center>   
<img src="refs/img_22.jpg" width="200" height="200">
<img src="heitz/result_22.jpg" width="200" height="200">
<img src="gonthier/result_22.jpg" width="200" height="200">
<img src="scale_1/result_22.jpg" width="200" height="200">
    
<img src="refs/img_25.jpg" width="200" height="200">
<img src="heitz/result_25.jpg" width="200" height="200">
<img src="gonthier/result_25.jpg" width="200" height="200">
<img src="scale_1/result_25.jpg" width="200" height="200">
    
<img src="refs/img_27.jpg" width="200" height="200">
<img src="heitz/result_27.jpg" width="200" height="200">
<img src="gonthier/result_27.jpg" width="200" height="200">
<img src="scale_1/result_27.jpg" width="200" height="200">
    
<img src="refs/img_30.jpg" width="200" height="200">
<img src="heitz/result_30.jpg" width="200" height="200">
<img src="gonthier/result_30.jpg" width="200" height="200">
<img src="scale_1/result_30.jpg" width="200" height="200">
    
<img src="refs/img_37.jpg" width="200" height="200">
<img src="heitz/result_37.jpg" width="200" height="200">
<img src="gonthier/result_37.jpg" width="200" height="200">
<img src="scale_1/result_37.jpg" width="200" height="200">
</p>





