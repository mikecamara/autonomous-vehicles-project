import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class AdvAttack:

    def __init__(self, kl, attack_freq=5, eps=0.2):
        self.kl = kl
        self.attack_freq = attack_freq
        self.attackCount = 0
        self.epsilon = eps
        print('with attacks!')
        print(self)
        print(kl)

    
    def adversarial_pattern(self, image, label):
        image = tf.cast(image, tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(image)
            prediction = self.kl.model(image)
            loss = tf.keras.losses.MSE(label, prediction)
        
        gradient = tape.gradient(loss, image)
        
        signed_grad = tf.sign(gradient)
        
        return signed_grad

    def run(self, img_arr, throttle):
        print("Im a bad bad malicious hacker running")
        print(self)
        print(img_arr)
        # if num_rec != None and num_rec % self.attack_freq == 0:
        
        if 0 == 0:

            print("Im a bad bad malicious hacker neveer")
            self.attackCount+=1
            image = img_arr.reshape((1,) + img_arr.shape)
            ang = self.kl.model.predict(image)
            
            perturbation = self.adversarial_pattern(image, ang).numpy()
            perturb = ((perturbation[0]*0.5 + 0.5)*255)-50
            adv_img = np.clip(img_arr + (perturb*self.epsilon), 0, 255)
            adv_img = adv_img.astype(int)
            print("Im a bad bad malicious hacker")
            return adv_img
        else: 
            print("Im a bad bad malicious hacker huuuuummmmm")
            return img

    # def __call__(self, img):
    #     print("Im a bad bad malicious hacker far ouuut")
    #     self.attackCount+=1
    #     image = img.reshape((1,) + img.shape)
    #     ang = self.kl.model.predict(image)

    #     perturbation = self.adversarial_pattern(image, ang).numpy()
    #     perturb = ((perturbation[0]*0.5 + 0.5)*255)-50
    #     adv_img = np.clip(img + (perturb*self.epsilon), 0, 255)
    #     adv_img = adv_img.astype(int)
    #     adv_ang = self.kl.model.predict(adv_img)
    #     return ang, adv_ang