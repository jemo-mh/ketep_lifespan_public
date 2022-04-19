import random
from unittest import result
from tqdm import tqdm
import pickle
import numpy as np


class cal_existing():
    def __init__(self):
        #variable for life span
        self.standard_life =30
        self.spent_year = random.randrange(1,38)
        self.left_over = self.standard_life - self.spent_year

        #variable for HI
        status = np.array([0,25,50,75,100])
        self.vis_scan=np.random.choice(status,1)
        self.corona = np.random.choice(status,1)
        self.ir = random.randrange(30,100)
        self.pd = np.random.choice(status,1)

        self.munjin = random.randrange(5,40)
        self.HI = self.cal_HI()

    def cal_HI(self):
        '''
        #TODO : Calculate HI(Health Index) based on KESCO algorithm
        calculate Status value with weight 
        '''
        self.ins1_ = round(self.vis_scan[0] * 0.12,2)   # 육안
        self.ins2_ = round(self.corona[0] * 0.28,2)     #코로나
        self.ins3_ = round(self.ir * 0.32,2)            #적외선
        self.ins4_ = round(self.pd[0] * 0.28,2)         # PD

        jindan = 0.6 * sum([self.ins1_, self.ins2_, self.ins3_, self.ins4_])
        HI = round(jindan + self.munjin,3)
        return HI

    def eval1(self):
        return self.left_over* (self.HI *0.01)
    
    def eval2(self):
        ret = (self.standard_life*(self.HI*0.01))*0.38 + self.left_over #spent_year is negative value
        return ret

    def eval3(self):
        ret = self.left_over + self.left_over*self.HI*0.01
        return ret

    def cal(self):
        # self.HI = self.cal_HI()
        if self.left_over < 10:
            if self.left_over < (self.standard_life/5):
                if self.HI >=61.5:
                    if self.spent_year > self.standard_life:  #기준수명경과
                        final = self.eval2()# cal 2
                    else:
                        final = self.eval3()#cal3
                else:
                    if self.HI<61.5 and self.HI>30:
                        if self.spent_year > self.standard_life:  #기준수명경과
                            final =0 #replace immediatly
                        else:
                            final = self.eval1()
                    else:
                        if self.HI <=30:
                            final = 0 # replace immediatly
            else:
                if self.HI >=30 :
                    final = self.eval1()
                else:
                    final =0
        else:
            if self.HI>=30:
                final = self.eval1()
            else:
                final=0     
        return round(final)

    def ret_result(self):
        #  육안검사, 코로나검사, 적외선검사, PD검사,문진점수, 기준수명, 경과수명, HI, 최종수명
        print("hi", self.HI)
        rt_list = [self.vis_scan[0], self.corona[0], self.ir, self.pd[0], self.munjin, self.standard_life,  self.spent_year,  self.HI ,self.cal()]
        return rt_list

def main():
    # f= open("datagen_lifespan.txt",'w')
    f= open("Dataset/Train.txt",'w')
    for i in tqdm(range(200000)):
        a = cal_existing()
        a = a.ret_result()
        f.write(",".join(str(j) for j in a)+"\n")
        # print(a)
    f.close()
    
    f= open("Dataset/Test.txt",'w')
    for i in range(50000):
        a = cal_existing()
        a = a.ret_result()
        f.write(",".join(str(j) for j in a)+"\n")
        # print(a)
    f.close()

main()
