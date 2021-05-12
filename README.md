Team Name: The BenchBotics.

Problem Statement:

Advancements in computer vision and machine learning technologies have transformed plant scientists' ability to incorporate high-throughput phenotyping into plant breeding. Detailed phenotypic profiles of individual plants can be used to understand plant growth under different growing conditions. As a result, breeders can make rapid progress in plant trait analysis and selection under controlled and semi-controlled conditions, thus accelerating crop improvements. Unfortunately, current commercially available platforms are large and very expensive. The upfront investment limits breeders use of high-throughput phenotyping in modern breeding programs. The expensive, state-of-the-art phenotyping solutions and labor-intensive activities involved in plant breeding motivated us to develop an open-source and low-cost solution for nondestructive greenhouse high-throughput phenotyping, the BenchBot (Figures 1 and 2).

![](https://lh3.googleusercontent.com/wy9g3Byqiq1p3fbQSD-Tx-LGWMejucPnNbrWeFl1wbBFmYIaRccxNdAIs86_ggRPCVKlxFESCcSEuplfygy6uMn5nr4mINTcfun_41CEPka1U-mjDLjkEX7y9NEVLsLc97CTJ_Fd)

Figure 1. Phenotyping prototype, RGB-D image, mounted camera and light-source.

![](https://lh4.googleusercontent.com/JZ53LfRkAU1_aVDWCOarq-r1bM9DKnWBmDWIJbCfs7PFn2ZB6KUxG2feGATFxga5OfLDOGVdPteXpgOynkX-JBJ-PU0r33f3yr-z7jwUwRFrtam8LI4DsEEYWFxuVUbzH1gWSJMy)

Figure 2. Parts of the phenotyping prototype (yellow boxes) and movements (blue arrows).

The BenchBot is a plant phenotyping platform which consists of two major components: a semi-automatic image acquisition device and a central process unit to control the platform and camera movement. The BenchBot is low-cost, modular, upgradeable, portable, and easy to use. Specifically, the BenchBot can be adjusted to work in benches of different dimensions and the camera height can be changed to accomodate plants of different heights and sizes. We are currently examining image acquisition and processing approaches that work best under greenhouse conditions(i.e. high levels of light and heat) (Figure 3). Furthermore, we are developing image processing algorithms needed to detect and identify plants, detect leaves, determine leaf area and estimate plant biomass.

![](https://lh3.googleusercontent.com/6jR6mMxGQx8r-bE_0iw8R42mwGKxvZ9-s4aUOHBtJMrOAqJf-m4lrBMP9LFphl1SruVAwgY50JtWsERhTAJ23uKyWOao8DPUOCu90nLlhfKRkZbZqZLpGUrH_HfcCU5AGuWg5O7q)![](https://lh3.googleusercontent.com/kbPNpJosM7XRCpAjtSYZyt_JaQQXPxW9Jmt0WmWoyPTzRIE-cYmvP89UyTWhPfBwyBngCQWMczkfFvUBr1nHNQgXV_gNQc6_1nnxkLcNiP-cCN9fSXA4yiUBzIe-Mly0g-3uGyoJ)![](https://lh4.googleusercontent.com/MRSuv2uKqI0O_2j5QxUdnnDfzoBR0IQuPRPX39eGryn693t08GQ_2MVbUpybM_nXpEr4Ylz0Gjnwl3QbGJaFObAIp28IGgsz0T2RuYsoaILIZIVBo1sGGdi3G4ccUAY_7LExNQdy)![](https://lh6.googleusercontent.com/LIZZnNR-4nKKFjsNjYdSZonyhbpNzqvWclnUym9wIfeYHy884ld48WE85LkEri94eJ8ZCumlyIW-yLOe5aEDAHZzQzbj_7krFpr3OZGnSy9YE798JSvrpphuzhV44JQUJQoj0j3X)

Figure 3. Example of images acquired under extreme conditions using the BenchBot and the IntelD535i camera.

Our first prototype consisted of a frame that can be moved along greenhouse benches with a mounted camera system (camera and lights) that stops at each potted plant to take individual RGB and depth images. We are currently developing the semi-autonomous system which requires minimal human intervention. Expected products include a stable control application and a suitable database that enables biomass measurements and plant detection algorithms. We are also working on full automation of the platform and image acquisition. We anticipate broad adoption of this technology among public and private plant breeders.

  

One of the biggest challenges that we face in the development of this platform is the management of data flow and data analysis in real time. The OAK-D camera could be a game changer to our project, since we could compress information that is transmitted to the cloud for further analyses, thus allowing for efficient real-time data flow and analysis. This proposal seeks to build on our prior work to create a semi-automated and non-invasive method for the precise measurement of plant species, biomass, and leaf area and number under greenhouse conditions, using the new OpenCV camera (OAK-D) and the Intel Distribution of OpenVINO Toolkit. The successful execution of this project will greatly increase access to high-throughput phenotyping solutions for breeders, thus accelerating crop improvements globally.

  

This proposal will be executed in 12 weeks as follows:

TASK 1 - Prototype preparation: Hardware adjustments. The OAK-D will be connected to a Raspberry Pi 4 and a 2400mA power bank with all elements in a single functional package housed in a weatherproof casing and will be mounted to the BenchBot. Following, we will test the camera under different lighting conditions and ensure that we do not have 3A issues (Auto-exposure, Auto-white-balancing, Auto-focus). Finally, we will calibrate the unit for high accuracy in depth detection of plants larger than 5 cm. Weeks 1 and 2.

TASK 2 - Data collection: Images will be acquired once a week for six consecutive weeks. Each week the OAK-D camera will take RGB images and depth images of 60 plants (20 broadleaves, 20 grasses, 20 clovers). We will correlate destructive biomass values with depth information for each species each week. Weeks 3, 4, 5, 6, 7, and 8.

TASK 3  - Preparation of the model for semantic segmentation for detecting broadleaves, grasses, and clovers: A segmentation model will be developed and trained using an existing image dataset. This model will be ported into the OpenVINO data flow to segment and classify image components as weeds and soil. Weeks 5, 6, 7, 8, and 9.

TASK 4 - Implementation of codes in DepthAI/OpenVINO/OpenCV: Code will be implemented directly using DepthAI and an inference model (developed in TASK 3) for the segmentation and classification of weeds. Segmentation and classification will draw on height and biomass measurements, a reference object, and the OpenCV library to produce output. Weeks 8 and 9.

TASK 5 - Testing and validation: Acquired images will be used to test the classification and detection of height and biomass of each plant. As a final step, the system will be validated with new images and real-time processing of plants under greenhouse conditions. Weeks 10, 11, and 12.

Furthermore, this camera system has broad applicability, it could be mounted to UAVs or tractors for the detection, identification, and mapping of plants using precision technologies to facilitate activities such as pest, nutrient, and water management.

We plan to do weekly development tracking and scripts documentation and testing through our GitHub repository [https://github.com/precision-sustainable-ag/Phenotyping-prototype/wiki](https://github.com/precision-sustainable-ag/Phenotyping-prototype/wiki). Our experience in the last competition, where we used the OAK-D camera for weed detection in extreme conditions, will serve as a base for the new approach proposed here. Moreover, the outcomes of this new competition will serve as a basis for a future Hackathon that the new North Carolina State University initiative ([Plant Science Initiative](https://cals.ncsu.edu/psi/)) plans to hold in the fall of this year.

Category: Agriculture.

Region: Region 1 - North America.

Team Type : University Team
