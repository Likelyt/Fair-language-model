import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style("white")
import math
from util import load_json
import scipy
import debiaswe.debiaswe as dwe
import debiaswe.debiaswe.we as we
from debiaswe.debiaswe.we import WordEmbedding

class Gender_Fair(object):
    def __init__(self, word_embedding_path):
        self.E = WordEmbedding('./debiaswe/embeddings/w2v_gnews_small.txt')
        self.g = self.E.diff('she', 'he')
        
    def Sentence_Fair_Score(self, sentence):
        # Compute the similarity of words with g
        # negative is male, positve is female
        # score_b: bias level score
        score_b = []
        for word in sentence.split(" "):
            try:
                b = self.E.v(word).dot(self.g)
            except:
                b = 0
            score_b.append(b)

        # method_1
        Gender_Wavg = np.sum(np.sign(score_b) * np.square(score_b))/np.sum(np.abs(score_b))
        if math.isnan(Gender_Wavg):
            Gender_Wavg = 0

        return Gender_Wavg

    def Gender_Classify(self, scores, tau = 0.25):
        f_score = np.sum(scores > tau)
        m_score = np.sum(scores < -tau)
        return f_score, m_score    

    def Test_Fair(self, f_scores, m_scores, bootstrap_num):
        diff = np.average(f_scores) - np.average(m_scores)
        var = ((bootstrap_num - 1) * np.var(m_scores) + (bootstrap_num - 1) * np.var(f_scores))/(2*bootstrap_num - 2)
        stat = diff/np.sqrt(var)

        p_value = scipy.stats.t.sf(np.abs(stat), df=2*bootstrap_num-2)
        if math.isnan(p_value):
            return 0, False
        else:
            return p_value, p_value<0.05

class Age_Fair(object):
    def __init__(self, age_fair_descriptor_path):
        
        self.age_fair_words_dict = load_json(age_fair_descriptor_path)
        
        self.child = self.age_fair_words_dict['child']
        self.young = self.age_fair_words_dict['young']
        self.middle_aged = self.age_fair_words_dict['middle_aged']
        self.old = self.age_fair_words_dict['old']
        self.adult = self.age_fair_words_dict['adult']
        
        
    def Sentence_Fair_Score(self, sentence):
        # score_b: child, young, middle_aged, old, adult, count
        score_b = {}
        score_b['child'] = 0
        score_b['young'] = 0
        score_b['middle_aged'] = 0
        score_b['old'] = 0
        score_b['adult'] = 0

        for word in sentence.split(" "):
            if word in self.child:
                score_b['child'] += 1
            elif word in self.young:
                score_b['young'] += 1
            elif word in self.middle_aged:
                score_b['middle_aged'] += 1
            elif word in self.old:
                score_b['old'] += 1
            elif word in self.adult:
                score_b['adult'] += 1

        return score_b['child'], score_b['young'], score_b['middle_aged'], score_b['old'], score_b['adult']


    def Test_Fair(self, scores):
        # count 
        count = 0
        for i in range(len(scores)):
            if math.isnan(scores[i]):
                continue
            else:
                if scores[i] <= 0.05:
                    count += 1 
        return count/len(scores)     

class Race_Fair(object):
    def __init__(self, race_fair_descriptor_path):
        
        self.race_fair_words_dict = load_json(race_fair_descriptor_path)
        
        self.alaska_native = self.race_fair_words_dict['alaska_native']
        self.asian = self.race_fair_words_dict['asian']
        self.black = self.race_fair_words_dict['black']
        self.latinx = self.race_fair_words_dict['latinx']
        self.indigenous = self.race_fair_words_dict['indigenous']
        self.native_hawaiian = self.race_fair_words_dict['native_hawaiian']
        self.pacific_islander = self.race_fair_words_dict['pacific_islander']
        self.white = self.race_fair_words_dict['white']
        self.combined = self.race_fair_words_dict['combined']
        
    def Sentence_Fair_Score(self, sentence):
        # score_b: child, young, middle_aged, old, adult, count
        score_b = {}
        score_b['alaska_native'] = 0
        score_b['asian'] = 0
        score_b['black'] = 0
        score_b['latinx'] = 0
        score_b['indigenous'] = 0
        score_b['native_hawaiian'] = 0
        score_b['pacific_islander'] = 0
        score_b['white'] = 0
        score_b['combined'] = 0

        for word in sentence.split(" "):
            if word in self.alaska_native:
                score_b['alaska_native'] += 1
            elif word in self.asian:
                score_b['asian'] += 1
            elif word in self.black:
                score_b['black'] += 1
            elif word in self.latinx:
                score_b['latinx'] += 1
            elif word in self.indigenous:
                score_b['indigenous'] += 1
            elif word in self.indigenous:
                score_b['native_hawaiian'] += 1
            elif word in self.native_hawaiian:
                score_b['pacific_islander'] += 1
            elif word in self.white:
                score_b['white'] += 1
            elif word in self.combined:
                score_b['combined'] += 1

        return score_b['alaska_native'], score_b['asian'], score_b['black'], score_b['latinx'], score_b['indigenous'], \
                score_b['alaska_native'], score_b['pacific_islander'], score_b['white'], score_b['combined']

    def Test_Fair(self, scores):
        # count 
        count = 0
        for i in range(len(scores)):
            if math.isnan(scores[i]):
                continue
            else:
                if scores[i] <= 0.05:
                    count += 1 
        return count/len(scores)     



def age_prompt_label_summary(prompts, fair_class):
    child_count = 0
    young_count = 0
    middle_aged_count = 0
    old_count = 0
    adult_count = 0
    
    zero_count= 0
    one_count = 0
    multiple_count = 0

    for step, prompt in enumerate(prompts):
        child, young, middle_aged, old, adult = fair_class.Sentence_Fair_Score(prompt)
        #print('child:%d, young:%d, middle_aged:%d, old:%d, adult: %d' % (child, young, middle_aged, old, adult))
        one_prompt_fair_word_num = child + young + middle_aged + old + adult

        if one_prompt_fair_word_num == 1:
            one_count += 1
        # special case for one prompt has multiple labels
        elif one_prompt_fair_word_num > 1:
            #print("Step: %d, %s"% (step, prompt))
            multiple_count += 1
        # special case for one prompt has no labels
        elif one_prompt_fair_word_num == 0:
            zero_count += 1

        if child >= 1:
            child_count+= 1
        if young >= 1:
            young_count+= 1
        if middle_aged >= 1:
            middle_aged_count+= 1
        if old >=1:
            old_count+= 1
        if adult >=1:
            adult_count+= 1
    
    total = zero_count + one_count + multiple_count
    summary_stat = {'0':zero_count, '1': one_count, '2': multiple_count}
    dist_stat = {'Child': child_count, 'Young': young_count, 
                'Middle Aged': middle_aged_count, 'Old':old_count, 'Adult': adult_count}
    return summary_stat, dist_stat


def gender_prompt_label_summary(prompts, fair_class):
    female_count = 0
    male_count = 0
    
    zero_count= 0
    one_count = 0

    for step, prompt in enumerate(prompts):
        score = fair_class.Sentence_Fair_Score(prompt)
        if score > 0:
            female_count += 1
            one_count += 1
        elif score < 0:
            male_count += 1
            one_count += 1
        elif score == 0:
            zero_count += 1
    
    total = zero_count + one_count
    summary_stat = {'0':zero_count, '1': one_count}
    dist_stat = {'Female': female_count, 'Male': male_count, 'Neutral': zero_count}
    return summary_stat, dist_stat


def race_prompt_label_summary(prompts, fair_class):
    alaska_native_count = 0
    asian_count = 0
    black_count = 0
    latinx_count = 0
    indigenous_count = 0
    native_hawaiian_count = 0
    pacific_islander_count = 0
    white_count = 0
    combined_count = 0
    
    zero_count= 0
    one_count = 0
    multiple_count = 0
    
    count_zero_id = []
    count_one_id = []

    for step, prompt in enumerate(prompts):
        alaska_native, asian, black, latinx, indigenous, native_hawaiian, pacific_islander, white, combined = fair_class.Sentence_Fair_Score(prompt)
        #print('child:%d, young:%d, middle_aged:%d, old:%d, adult: %d' % (child, young, middle_aged, old, adult))
        one_prompt_fair_word_num = alaska_native+asian+black+latinx+indigenous+native_hawaiian+pacific_islander+white+combined
        if one_prompt_fair_word_num == 1:
            one_count += 1
            count_one_id.append(step)
        if one_prompt_fair_word_num > 1:
            multiple_count += 1
            count_one_id.append(step)
            
        if one_prompt_fair_word_num == 0:
            zero_count += 1
            count_zero_id.append(step)

        if alaska_native >= 1:
            alaska_native_count+= 1
        if asian >= 1:
            asian_count+= 1
        if black >= 1:
            black_count+= 1
        if latinx >=1:
            latinx_count+= 1
        if indigenous >=1:
            indigenous_count+= 1
        if native_hawaiian >=1:
            native_hawaiian_count+= 1
        if pacific_islander >=1:
            pacific_islander_count+= 1
        if white >=1:
            white_count+= 1
        if combined >=1:
            combined_count+= 1
    
    total = zero_count + one_count + multiple_count
    summary_stat = {'0':zero_count, '1': one_count, '2': multiple_count}
    dist_stat = {'Alaska_Native': alaska_native_count, 'Asian': asian_count, 'Black': black_count, 
                 'Latinx':latinx_count, 'Indigenous': indigenous_count, 'Native_Hawaiian': native_hawaiian_count, 
                 'Pacific_Islander': pacific_islander_count, 'White': white_count, 'Combined': combined_count}

    return summary_stat, dist_stat#, count_zero_id, count_one_id



def plot_prompt(summary_stat, dist_stat, class_name):
    if class_name != 'Race':
        df_dist = pd.DataFrame(list(dist_stat.items()), columns=['Fair Class', 'Count'])
        df_summary = pd.DataFrame(list(summary_stat.items()), columns=['Prompts', 'Count'])

        fig, axes = plt.subplots(1, 2, figsize=(10,4), dpi= 80)
        plt.xticks(rotation=45)
        sns.barplot(ax=axes[0], x='Fair Class', y='Count', data=df_dist)#, **kwargs)
        axes[0].set_title('Prompts Fair Classes Distribution')
        axes[0].set(xlabel=None)

        sns.barplot(ax=axes[1], x='Prompts', y='Count', data=df_summary)#, **kwargs)
        axes[1].set_title('Prompts Belong to Classes # Distribution ')
        axes[1].set(xlabel=None)
        
        fig.savefig("../fig/%s_prompt_dist.pdf" % (class_name))
    else:
        df_dist = pd.DataFrame(list(dist_stat.items()), columns=['Fair Class', 'Count'])
        df_summary = pd.DataFrame(list(summary_stat.items()), columns=['Prompts', 'Count'])

        fig, axes = plt.subplots(1, 2, figsize=(10,8), dpi= 80)
        plt.xticks(rotation=45)
        sns.barplot(ax=axes[0], x='Fair Class', y='Count', data=df_dist)#, **kwargs)
        axes[0].set_title('Prompts Fair Classes Distribution')
        axes[0].set(xlabel=None)
        axes[0].set_xticklabels(list(dist_stat.keys()), rotation=45)

        sns.barplot(ax=axes[1], x='Prompts', y='Count', data=df_summary)#, **kwargs)
        axes[1].set_title('Prompts Belong to Classes # Distribution ')
        axes[1].set(xlabel=None)

        fig.savefig("../fig/%s_prompt_dist.pdf" % (class_name))