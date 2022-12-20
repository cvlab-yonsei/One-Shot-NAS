# Complexity Aware Supernet Training for One-Shot NAS

## Pre-requisites
This repository has been tested with the following libraries:
* Python (3.8.8)
* Pytorch (1.8.1)

## Preparation and Download
1. Download NAS-Bench-201 API: [`NAS-Bench-201-v1_1-096897.pth`](https://drive.google.com/open?id=16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_) (4.7G) and save it under ./Complexity_Aware_Supernet_Training/data/ folder.

2. Install NasBench201 via pip. 
```
pip install nas-bench-201
```

Please refer to [AutoDL-Projects](https://github.com/D-X-Y/AutoDL-Projects/blob/main/docs/NAS-Bench-201.md) for more details.

## Acknowledgements
Our codes are partly based on the following repositories.
- [AutoDL](https://github.com/D-X-Y/AutoDL-Projects)
- [GM-NAS](https://github.com/skhu101/GM-NAS)
