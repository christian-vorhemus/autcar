
# Use Mirosoft Custom Vision to train a model

In this tutorial we will learn how to use the customvision.ai online service to train a model. This works completely without code, the trained model can be downloaded as a file. We aim to create a traffic sign detector: Our car should be capable of recongizing two different traffic signs:

<p float="left">
  <img src="../images/major_road_sign.jpg" width="200" />
  <img src="../images/stop_sign.jpg" width="200" /> 
</p>

## Why using Custom Vision?

We can train and execute our model locally without any other services. However, very often, especially when dealing with large training or when large computing power is required, cloud services are used. Additionally, there is a variety of services that offer a zero-code interface to train models. Custom Vision is one of them. With Custom Vision you can simply upload images, add a label and train a model. Custom Vision makes use of **transfer learning** meaning that a base model was already trained on million of images and the already learned model weights are fine tuned with the images you add. 

## Create an Azure account and configure Custom Vision

1) You need an Azure subscription to use Custom Vision. If you don't have one yet, you can start with a free account [here](https://azure.microsoft.com/free/). You have to enter some information including an email address which you can then use to sign in.

2) Go to [www.customvision.ai](https://www.customvision.ai) and sign in with the email address you just provided. Accept the conditions and you should see the empty start page

  <img src="../images/customvision_1.png" width="400">

3) Click on "New Project". Add a name and create a new Resource Group. When adding a new Resource Group you have to provide a name, a subscription you want to use and the location where your resource group is placed.

4) Now you should be able to add additional configurations for your project: Choose "Classification" as a project type and "Multiclass" as the classification type. For the domain, choose a **compact** one (only compact models can be exported), let's just take "General (compact)" and for the "Export Capabilities" select "Basic platforms". Create the project.

## Create an upload training data

