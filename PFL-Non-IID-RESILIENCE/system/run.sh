#!/bin/bash
@echo off
conda activate fl
#com entropia
#python main.py -nc 100 -nb 10 -c cka -cp 100 -jr 0.2 -ws 1 -ncl 18 -ent 1 -gr 30 -t 5
#sem entropia mas com pesos
#python main.py -nc 100 -nb 10 -c cka -cp 100 -jr 0.2 -ws 1 -ncl 18 -wc 1 -gr 30 -t 5
#Sem pesos e sem entropia
#python main.py -nc 100 -nb 10 -cp 100 -jr 0.2 -ws 0 -gr 50 -t 5

#python main.py -nc 60 -nb 10 -gr 30 -c cka -cp 100  -t 5 -jr 0.3 -data "mnist" -ncl 10 -tsa A
#python main.py -nc 60 -nb 10 -gr 30 -c cka -cp 100  -t 5 -jr 0.3 -data "mnist" -ncl 10 -tsa D
#python main.py -nc 60 -nb 10 -gr 30  -t 5 -jr 0.3 -data "mnist"

############################################################################################################


#python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 1 -wc 0 -ft 0.01 -t 10
python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 1 -wc 0 -ft 0.05 -t 10 -data Cifar10

#python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 1 -ft 0.01 -t 10
python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 1 -ft 0.05 -t 10 -data Cifar10

#python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 0 -ft 0.01 -t 10
python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 0 -ft 0.05 -t 10 -data Cifar10

############################################################################################################
"""

python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 1 -wc 0 -ft 0.01 -t 10 -algo MOON
python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 1 -wc 0 -ft 0.05 -t 10 -algo MOON

python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 1 -ft 0.01 -t 10 -algo MOON
python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 1 -ft 0.05 -t 10 -algo MOON

python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 0 -ft 0.01 -t 10 -algo MOON
python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 0 -ft 0.05 -t 10 -algo MOON

############################################################################################################

python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 1 -wc 0 -ft 0.01 -t 10 -algo FedALA
python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 1 -wc 0 -ft 0.05 -t 10 -algo FedALA

python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 1 -ft 0.01 -t 10 -algo FedALA
python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 1 -ft 0.05 -t 10 -algo FedALA

python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 0 -ft 0.01 -t 10 -algo FedALA
python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 0 -ft 0.05 -t 10 -algo FedALA
"""
