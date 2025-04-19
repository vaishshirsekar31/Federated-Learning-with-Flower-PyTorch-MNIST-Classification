# Federated-Learning-with-Flower-PyTorch-MNIST-Classification
This project demonstrates federated learning (FL) on the MNIST dataset using the Flower framework and PyTorch. It enables collaborative model training across multiple clients without sharing raw data, preserving privacy while achieving high accuracy.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Flower](https://img.shields.io/badge/Flower-1.0%2B-green)

A privacy-preserving implementation of federated learning for MNIST handwritten digit classification.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [References](#references)

## Overview
This project demonstrates federated learning on the MNIST dataset using:
- Flower framework for federated coordination
- PyTorch for model training
- Achieves 99.3% accuracy in 5 rounds without centralized data collection

## Key Features
- 🛡️ **Privacy-first**: No raw data leaves client devices
- ⚡ **Efficient**: Only model updates are shared (~1MB/round)
- 📈 **High accuracy**: 99.3% test accuracy
- 🔧 **Easy to extend**: Add more clients or datasets

## Installation
```bash
git clone https://github.com/yourusername/federated-mnist.git
cd federated-mnist

## Run with 3 clients for 5 rounds:
python federated_mnist.py --num-clients 3 --rounds 5
