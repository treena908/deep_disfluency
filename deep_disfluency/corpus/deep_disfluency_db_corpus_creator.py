import json
import argparse
import re
import os
import sys
from collections import deque
from copy import deepcopy
from corpus_util import detection_corpus_format
from corpus_util import strip_disf_tags_from_easy_read
from corpus_util import get_edit_terms_from_easy_read
from corpus_util import easy_read_disf_format
from corpus_util import disf_tags_from_easy_read
from ast import literal_eval
import pandas as pd
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
tags={}
index_map={}
count_error={}
count_map={}
ranges=[]
valid_words=[]
run=True
test=False
pos_test=False
def print_tag(wordList, DisfluencytagList,indexList):
    print('****************************************')
    print('final tags')
    print(len(wordList))
    print(len(DisfluencytagList))
    for word,tag in zip(wordList, DisfluencytagList):
        print(word+ " :"+tag)
def index_mapping(tokens):

    count=0
    pattern1 = re.compile('\[x')
    pattern2 = re.compile('\d+\]')
    pattern3=re.compile(r"^&?[-]?[a-zA-Z]+$|^[a-zA-Z]*[\'][a-zA-Z]+$|^<[a-zA-Z]+$|^[a-zA-Z]>$|^<[a-zA-Z]+>$")

    for i,words in enumerate(tokens):

        if pattern3.match(words):
        # if words!='[//]' and words!='[/]' and words!='<' and  words!='>' and words!=',' and word and  not pattern1.match(words) and not pattern2.match(words):
            index_map[str(i)] = count
            if '<' in words:
                valid_words.append(words[1:])
            elif '>' in words:
                valid_words.append(words[:-1])
            elif ',' in words:
                valid_words.append(words[:-1])
            else:
                valid_words.append(words)



            count+=1

    # return map
def write_edit_term_corpus(testcorpus, targetfilename, debug=False):
    """Write a file cleaned of reparanda and edit terms from a test corpus.

    Keyword Arguments:
    testcorpus -- a string separated by newline markers which
    has all the disfluency markup.
    targetfilename -- a string giving the location of
    the edit term filled corpus.
    """
    print("Writing edit term corpus...")
    edit_term_corpus = open(targetfilename , "w")
    for line in testcorpus.split("\n"):
        if line == "":
            continue
        split = line.split(",")
        # no need to write the indices to source data
        if split[0] == "REF":
            continue
        elif split[0] == "POS":
            pos = split[1]
        else:
            uttref = split[0]
            text = split[1]
            continue
        editterm_examples = get_edit_terms_from_easy_read(text, pos)
        if editterm_examples != []:
            for my_editterm, my_poseditterm in editterm_examples:
                edit_term_corpus.write(
                    uttref + "," + my_editterm + "\nPOS," +
                    my_poseditterm + "\n")
    edit_term_corpus.close()
    print("done")
    return
def write_clean_corpus(testcorpus, targetfilename):
    """Write a file cleaned of reparanda and edit terms from a test corpus.

    Keyword Arguments:
    testcorpus -- a string separated by newline markers which
    has all the disfluency markup.
    targetfilename -- a string giving the location of the cleaned corpus.
    """
    print("Writing clean corpus...")
    clean_corpus = open(targetfilename , "w")
    for line in testcorpus.split("\n"):
        # print line
        if line == "":
            continue
        split = line.split(",")
        # no need to write the indices to source data
        if split[0] == "REF":
            continue
        elif split[0] == "POS":
            pos = split[1]
        else:
            uttref = split[0]
            text = split[1]
            continue

        words = strip_disf_tags_from_easy_read(text)
        pos_tags = strip_disf_tags_from_easy_read(pos)
        disfluencies = disf_tags_from_easy_read(text)
        clean_word_string = ""
        clean_pos_string = ""
        for i in range(0, len(disfluencies)):
            if "<e" in disfluencies[i] or "<rm" in disfluencies[i] \
                    or "<i" in disfluencies[i]:
                continue
            clean_word_string += words[i] + " "
            clean_pos_string += pos_tags[i] + " "
        clean_word_string = clean_word_string.strip()
        clean_pos_string = clean_pos_string.strip()
        if clean_word_string == "":
            continue
        clean_corpus.write(
            uttref + "," + clean_word_string + "\nPOS," +
            clean_pos_string + "\n")
    clean_corpus.close()
    print ("done")
    return
def check_contraction(tokens,ind):
    contraction_pattern = re.compile('\'[a-zA-Z]+')
    if contraction_pattern.match(tokens[ind - 1]):
        prev = 1
        prev_tag=False
        if ind - 1 < 0 or str(ind - 1) not in index_map.keys():
            print('inside contraction')
            track_disfluency_type('problem')
            track_error('problem', trans_name, utt_count)
            prev_tag = True

        while prev <= 2 and ind-prev>=0:
            tag = ""
            temp = ""
            if prev == 2:
                temp = temp + '<rms id="{}"/>'
            else:
                temp = temp+ '<rm id="{}"/>'
            if ind - prev >= 0 and str(ind-prev) in tags.keys():
                tag = tags[str(ind - prev)]

            if not prev_tag:
                tag += temp.format(index_map[str(ind - 1)])
            tags[str(ind - prev)] = tag  # tagging reparandum of rep. word
            prev += 1
        return True
    return False
        # word contraction tagging end
def track_disfluency_type(name):
    if name in count_map.keys():
        count_map[name] += 1

    else:
        count_map[name]=1
def track_error(key,trans_name,utt_count):

    if key in count_error.keys():
        count_error[key] += trans_name+":"+str(utt_count)+"\n"

    else:
        count_error[key]=trans_name+":"+str(utt_count)+"\n"
def read_file_ranges(range_files):
    ranges.clear()
    if range_files:
        for range_file_name in range_files:
            rangeFile = open(range_file_name, "r")
            for line in rangeFile:
                a = line.strip("\n")
                ranges.append(a)  # conversation_no
                # print a
            rangeFile.close()
        print("files in ranges =%s " % (str(len(ranges))))
        #print "files in ranges = " + str(len((ranges))
def extract_pos(pair,inv_tag):
  pos_list=[]
  if inv_tag:
    pair=pair[1:]

  count=0
  jump=False
  print('len pair: '+str(len(pair)))
  for p in pair:

      if jump:
          jump=False
          pos_list[len(pos_list)-1]=pos_list[len(pos_list)-1]+p[1]
          count+=1
          continue
      if p[0] == ',':
          continue

      if count<len(valid_words):
          if p[0]==valid_words[count]:
            # print('milse: '+p[0]+" "+valid_words[count])
            pos_list.append(p[1])
            count+=1
          else:
              if '&' in valid_words[count] and p[0]==valid_words[count][1:]:
                  # print('milse: ' + p[0] + " " + valid_words[count])
                  pos_list.append(p[1])
                  count += 1
                  continue


              print('mile nai: ' + p[0] + " " + valid_words[count])
              pattern=re.compile(r"^[a-zA-Z]*[\'][a-zA-Z]+$")
              if pattern.match(valid_words[count]):
                  pos_list.append(p[1])
                  jump=True
      else:
          return pos_list,'valid_word list index out of range'




  return pos_list,'okay'
def print_anomaly(pos_tag,index_map,tokens):
    print('index_map')
    print(len(index_map))
    print(len(valid_words))
    print(valid_words)
    # for key in index_map.keys():
    #     print(tokens[int(key)])
    print(index_map)
    print(pos_tag)
    print(len(pos_tag))
    for p in pos_tag:
        print(p[0])

def clean_tokens(tokens):
    inv_tag=False
    if 'INV' in tokens:
        tokens.remove('INV')
        inv_tag = True
    if ',' in tokens:
        tokens.remove(',')
    return tokens,inv_tag

def make_DB_corpus(writeFile,writecleanFile,writeeditFile,target,filename,range_files):
    #samples = ['the &m &uh mother is [//] &um <I \'m assuming it \'s a mother > [//] is stepping in it']
    transcripts=[]
    if writeFile:
        df=pd.read_csv(THIS_DIR+'/../../DisfluencyProject/data_w_pos.csv')
        transcripts=df.filename.unique()
    else:
        transcripts.append('1')




    samples=[
        # 'mother standing in the overflowed water'
        'and <stand up by> [//] stand up <in the> [//] in a window is [/] is [/] <is over the> [/] &uh is over the sink'
        # 'the &m &uh mother is [//] &um < I \'m assuming it \'s a mother > [//] is stepping in it',
    # '<lot of> [/] a lot &uh of people do that',
    #  '<i can \'t> [//] I &uh don \'t do that',
    #    'he \'d [//] she did go home and lie down and rest and get some things changed',
    #          'and &uh <outside the> [//] &cup the cookie jar would have to be in the cupboard',
    # 'and there are dishes [//] &uh &uh two cups and a saucer on the sink',
    # 'and &uh &uh the [/] &uh a the [//] outside the window there"s a path leading to a garage it looks like',
    # '<her brother is> [/] her brother is taking cookies out_of a jar',
    # '&um <they \'re grading> [//] &uh they [/] they are going to &um get get get get [x 4] some cookies from the cookie jar',
    # 'and &th &th this [/] this is +...',
    # 'he \'s gonna [/] gonna fall because his [//] &uh the [/] <the cookies jar or> [//] <the the the [x 3] bench> [//] the &s four legged stool <whatever it is> [//] is [/] is gonna fall overwith him and the cookie jar'
             ]
    POS_list=[
        [('and', 'CC'), ('stand', 'VB'), ('up', 'RP'), ('by', 'IN'), ('stand', 'VB'), ('up', 'RP'), ('in', 'IN'),
         ('the', 'DT'), ('in', 'IN'), ('a', 'DT'), ('window', 'NN'), ('is', 'VBZ'), ('is', 'VBZ'), ('is', 'VBZ'),
         ('over', 'IN'), ('the', 'DT'), ('uh', 'UH'), ('is', 'VBZ'), ('over', 'IN'), ('the', 'DT'), ('sink', 'NN')]
        # [('mother', 'NN'), ('standing', 'VBG'), ('in', 'IN'), ('the', 'DT'), ('overflowed', 'JJ'), ('water', 'NN')]
        # [('and', 'CC'), ('stand', 'VB'), ('up', 'RP'), ('by', 'IN'), ('stand', 'VB'), ('up', 'RP'), ('in', 'IN'), ('the', 'DT'), ('in', 'IN'), ('a', 'DT'), ('window', 'NN'), ('is', 'VBZ'), ('is', 'VBZ'), ('is', 'VBZ'), ('over', 'IN'), ('the', 'DT'), ('uh', 'UH'), ('is', 'VBZ'), ('over', 'IN'), ('the', 'DT'), ('sink', 'NN')]
    ]
    overallWordsList = []  # all lists of words
    overallPOSList = []  # all lists of corresponding POS tags
    overallTagList = []  # all lists of corresponding Disfluency Tags
    # all list of corresponding index tags back to the original
    # transcription
    overallIndexList = []
    uttList = []
    trans_count=0
    ranges = []
    # get the ranges of the source corpus to extract data from
    if range_files:
        for range_file_name in range_files:
            rangeFile = open(range_file_name, "r")
            for line in rangeFile:
                a = line.strip("\n")
                ranges.append(a)  # conversation_no
                # print a
            rangeFile.close()
        print("files in ranges =%s "  %(str(len(ranges))))
    else:
        ranges.append('1')
    for trans_name in transcripts:
        if writeFile:
            if trans_name not in ranges:
                continue
            rows=df[df['filename']==trans_name]
            samples=rows.cleaned_text_w_disfluency_markers
            tags_pos = rows.pos_tag
        if pos_test:
            tags_pos=POS_list


        utt_count=0


        space = re.compile('[\s]+')
        for utt,tagging in zip(samples,tags_pos):

        # for utt in samples:
            print('utt: '+utt)
            tag_error = False
            inv_tag=False
            wordList = []
            POSList = []
            disfluencyTagList = []
            indexList = []
            tokens=utt.split()
            # print('tokens: '+str(tokens))
            tokens,inv_tag=clean_tokens(tokens)


            # print('tokens')
            # print(tokens)

            if len(tokens)==0:
                track_disfluency_type('empy_utt')
                track_error('empy_utt',trans_name,utt_count)
                continue
            # print(tokens)
            # index_map=index_mapping(tokens)
            index_mapping(tokens)
            # print('after index tokens: ' + str(tokens))
            # print('printing index map')
            # print(valid_words)
            if writeFile:
                pos_list = literal_eval(tagging)
                tag_pos,msg = extract_pos(pos_list, inv_tag)
                if msg!='okay':
                    track_disfluency_type('valid_word list index out of range')
                    track_error('valid_word list index out of range', trans_name, utt_count)
            elif pos_test:
                # tag_pos = tagging
                tag_pos,msg=extract_pos(tagging, inv_tag)


            else:
                tag_pos = ['none']

            if(len(tag_pos)!=len(valid_words)):
                print("pos_len_mismatch")
                print('tag_pos len :'+str(len(tag_pos)))
                print('valid word len: '+str(len(valid_words)))
                print_anomaly(tag_pos,valid_words,tokens)
                # continue

                track_disfluency_type("pos_len_mismatch")
                track_error("pos_len_mismatch",trans_name,str(utt_count))
                tag_error = True

            # print('index_map')
            # print(index_map)
            # print('before phrase ret.:' + str(tokens))
            retracing= [i for i, x in enumerate(tokens) if x == "[//]"]
            if len(retracing)>0:  #retacing tags
                for ind in retracing:
                    if ind>0 and '>' in tokens[ind-1]:#phrase retrace
                        #print("phrase retrace")
                        reparandum_stack = deque([])
                        idx = ind - 1
                        mark=ind-1
                        if len(tokens[ind-1])==1:
                            print('agerta')
                            mark=ind-2

                        track_disfluency_type("phrase retrace")
                        prev_tag=False
                        if mark < 0 or str(mark) not in index_map.keys():
                            track_disfluency_type('problem')
                            track_error('problem', trans_name, utt_count)
                            prev_tag = True
                        # reparandum tagging
                        digit_pattern = re.compile('\d+]')
                        while idx>=0 and  not space.match(tokens[idx]):
                            tag = ""

                            if '<' in tokens[idx] and len(tokens[idx]) > 1:
                                print('edge token with < :' + tokens[idx])
                                tag = ""
                                if idx >= 0 and idx < len(tokens) and str(idx) in tags.keys():
                                    tag = tags[str(idx)]
                                if mark < 0 or str(mark) not in index_map.keys():
                                    track_disfluency_type('problem')
                                    track_error('problem', trans_name, utt_count)
                                else:
                                    tag += '<rms id="{}"/>'.format(index_map[str(mark)])
                                tags[str(idx)] = tag
                                # adding onset of repeted phrase to stack
                                reparandum_stack.append(tokens[idx][1:])
                                break
                            elif '<' in tokens[idx] and len(tokens[idx]) == 1:
                                #the retrace phrase has space after < and then the token
                                print('edge token with only < :' + tokens[idx] +"  "+tokens[idx+1])
                                tag = ""
                                if idx+1 >= 0 and idx+1 < len(tokens) and str(idx+1) in tags.keys():
                                    tag = tags[str(idx+1)]
                                if mark < 0 or str(mark) not in index_map.keys():
                                    track_disfluency_type('problem')
                                    track_error('problem', trans_name, utt_count)
                                else:
                                    tag=re.sub('\<rm id=\"\d+\"\/\>', '', tag)
                                    tag += '<rms id="{}"/>'.format(index_map[str(mark)])
                                tags[str(idx+1)] = tag
                                # adding onset of repeted phrase to stack
                                reparandum_stack.pop()
                                reparandum_stack.append(tokens[idx+1])
                                break

                            elif not digit_pattern.match(tokens[idx]) and tokens[idx] not in ['[x']:
                                if idx >= 0 and idx < len(tokens) and str(idx) in tags.keys():
                                    tag = tags[str(idx)]
                                if not prev_tag:
                                    tag += '<rm id="{}"/>'.format(index_map[str(mark)])
                                tags[str(idx)] = tag
                                # adding repeted phrase to stack
                                if '>' in tokens[idx]:
                                    reparandum_stack.append(tokens[idx][:-1])
                                else:
                                    reparandum_stack.append(tokens[idx])

                            idx -= 1


                        stack_size = len(reparandum_stack)
                        #print('stack before')
                        #print(reparandum_stack)

                        # tagging reparandum of phrase repetition ends

                        # integrenum tagging
                        idx = ind + 1
                        prev_idx = mark
                        prev_tag=False
                        pattern = re.compile('&[-]+[a-zA-Z]+')
                        if mark <0  and str(mark) not in index_map.keys():
                            track_disfluency_type('problem')
                            track_error('problem', trans_name, utt_count)
                            prev_tag=True
                        while idx<len(tokens) and pattern.match(tokens[idx]) :
                            track_disfluency_type("integranum_phrase_retrace")
                            tag = ""
                            if idx < len(tokens) and str(idx) in tags.keys():
                                tag = tags[str(idx)]
                            if not prev_tag:
                                tag += '<i id="{}"/>'.format(index_map[str(prev_idx)])
                            tag += '<e/>'
                            tags[str(idx)] = tag
                            idx += 1
                        # integrenum tagging ends
                        # repair part of phrase repetition tagging
                        digit_pattern = re.compile('\d+]')
                        while idx<len(tokens) and len(reparandum_stack) > 0 :
                            tag = ""


                            if pattern.match(tokens[idx]):
                                #print('hesitation in repair part of phrase rep.')
                                track_disfluency_type("hesitation_repair_retrace")
                                if idx < len(tokens) and str(idx) in tags.keys():
                                    tag = tags[str(idx)]

                                tag += '<e/>'
                                tags[str(idx)] = tag

                            elif tokens[idx] not in ['[//]','[/]','[x'] and not space.match(tokens[idx]) and not digit_pattern.match(tokens[idx]):

                                if len(reparandum_stack) == stack_size:
                                    print(tokens[idx])
                                    # first word of repair rep.
                                    current_word = reparandum_stack.pop()
                                    # print('first')
                                    # print(prev_tag)
                                    if idx < len(tokens) and str(idx) in tags.keys():

                                        tag = tags[str(idx)]
                                    if not prev_tag:
                                        print(tokens[idx])
                                        tag += '<rps id="{}"/>'.format(index_map[str(prev_idx)])
                                    tags[str(idx)] = tag
                                    print(tags[str(idx)])
                                elif len(reparandum_stack) == 1:
                                    # print(tokens[idx])
                                    # print('last')
                                    print(tokens[idx])
                                    # last word of repair rep.
                                    current_word = reparandum_stack.pop()

                                    if idx < len(tokens) and str(idx) in tags.keys():
                                        print(tokens[idx])
                                        print('mid')
                                        tag = tags[str(idx)]
                                    if not prev_tag:
                                        tag += '<rpndel id="{}"/>'.format(index_map[str(prev_idx)])
                                    tags[str(idx)] = tag
                                else:
                                    # mid word of repair rep.
                                    current_word = reparandum_stack.pop()
                                    # print('mid')
                                    # print(tokens[idx])
                                    if idx < len(tokens) and str(idx) in tags.keys():
                                        tag = tags[str(idx)]

                                    else:
                                        if not prev_tag:
                                            tag += '<rp id="{}"/>'.format(index_map[str(prev_idx)])
                                        tags[str(idx)] = tag
                            idx += 1
                        #print('stack after')
                        #print(reparandum_stack)
                        if len(reparandum_stack) == 0:
                            print("end of repair of phrase ret. tagging")
                        else:
                            #print("error in repair of phrase ret. tagging")
                            tag=""
                            idx-=1
                            if idx < len(tokens) and str(idx) in tags.keys():
                                tag = tags[str(idx)]
                            if not prev_tag:
                                tag = re.sub('\<rp id=\"\d+\"\/\>', '', tag)
                                tag += '<rpndel id="{}"/>'.format(index_map[str(prev_idx)])
                            tags[str(idx)] = tag
                            track_disfluency_type('ret._repair_error')
                            track_error('ret._repair_error',trans_name,utt_count)
                    else:
                        #print("word retrace")
                        # tagging reparandum of retrace word
                        track_disfluency_type("word_retrace")
                        tag=""
                        #chech contraction in reparandum of retr. word

                        if check_contraction(tokens,ind):
                            track_disfluency_type("contraction_word_retrace")

                            #print('contraction in word retrace found')
                        # chech contraction in reparandum of retr. word ends
                        else:
                            #normal reparandum in word ret
                            if ind-1>=0 and str(ind-1) in tags.keys():
                                tag =  tags[str(ind-1)]

                            if ind-1<0 or str(ind-1) not in index_map.keys():
                                print('problem')
                                track_disfluency_type('problem')
                                track_error('problem',trans_name,utt_count)
                                # print(utt)
                                # print(tokens)
                                # print(index_map)
                                # print(ind-1)
                                # print(tokens[ind-1])
                            else:
                                tag+='<rms id="{}"/>'.format(index_map[str(ind-1)])
                            tags[str(ind - 1)]=tag # tagging reparandum of retrace word
                                # tagging normal reparandum of retrace word ends
                        #check if any integranum is after the retraced word
                        idx=ind+1
                        prev_idx=ind-1
                        pattern = re.compile('&[-]+[a-zA-Z]+')
                        prev_tag=False
                        if prev_idx < 0 or str(prev_idx) not in index_map.keys():
                            track_disfluency_type('problem')
                            track_error('problem', trans_name, utt_count)
                            prev_tag=True
                        while idx<len(tokens) and pattern.match(tokens[idx]) :
                            tag = ""
                            track_disfluency_type("integranum_word_retrace")
                            if idx <len(tokens) and str(idx) in tags.keys():
                                tag = tags[str(idx)]
                            if not prev_tag:
                                tag += '<i id="{}"/>'.format(index_map[str(prev_idx)])
                            tag += '<e/>'
                            tags[str(idx)] = tag
                            idx+=1
                        # check if any integranum is after the retraced word ends

                        # tagging repair of retrace word
                        tag=""

                        if idx<len(tokens) and str(idx+1) in tags.keys():
                            tag =  tags[str(idx+1)]
                        if ind-1 < 0 or str(ind-1) not in index_map.keys():
                            track_disfluency_type('problem')
                            track_error('problem', trans_name, utt_count)
                        else:
                            tag+='<rps id="{}"/>'.format(index_map[str(ind-1)])
                            tag += '<rpndel id="{}"/>'.format(index_map[str(ind-1)])
                        tags[str(idx)] = tag  # tagging repair of retrace word

                        # tagging repair of retrace word ends



            # print('before phrase rep.:'+ str(tokens))
            repetition = [i for i, x in enumerate(tokens) if x == "[/]"]
            if len(repetition)>0:  #repetition tags
                for ind in repetition:
                    if ind>0 and '>' in tokens[ind-1]:#phrase repetition

                        # print("phrase repetition" +str(ind) +" "+tokens[ind-1])
                        # print(str(tokens))
                        track_disfluency_type("phrase_repetition")
                        reparandum_stack = deque([])
                        idx=ind-1
                        mark=idx
                        if len(tokens[ind-1])==1:
                            mark=ind-2

                        while idx>=0 and idx<len(tokens)  and not space.match(tokens[idx]) :
                            tag = ""
                            if '<' in tokens[idx] and len(tokens[idx]) > 1:
                                print('edge token with < :' + tokens[idx])
                                tag = ""
                                if idx >= 0 and idx < len(tokens) and str(idx) in tags.keys():
                                    tag = tags[str(idx)]
                                if mark < 0 or str(mark) not in index_map.keys():
                                    track_disfluency_type('problem')
                                    track_error('problem', trans_name, utt_count)
                                else:
                                    tag += '<rms id="{}"/>'.format(index_map[str(mark)])
                                tags[str(idx)] = tag
                                # adding onset of repeted phrase to stack
                                reparandum_stack.append(tokens[idx][1:])
                                break
                            elif '<' in tokens[idx] and len(tokens[idx]) == 1:
                                #the rep. phrase has space after < and then the token
                                print('edge token with only < :' + tokens[idx] +"  "+tokens[idx+1])
                                tag = ""
                                if idx+1 >= 0 and idx+1 < len(tokens) and str(idx+1) in tags.keys():
                                    tag = tags[str(idx+1)]
                                if mark < 0 or str(mark) not in index_map.keys():
                                    track_disfluency_type('problem')
                                    track_error('problem', trans_name, utt_count)
                                else:
                                    tag=re.sub('\<rm id=\"\d+\"\/\>', '', tag)
                                    tag += '<rms id="{}"/>'.format(index_map[str(mark)])
                                tags[str(idx+1)] = tag
                                # adding onset of repeted phrase to stack
                                reparandum_stack.pop()
                                reparandum_stack.append(tokens[idx+1])
                                break
                            else:
                                if idx > 0 and idx<len(tokens) and str(idx) in tags.keys():
                                    tag = tags[str(idx)]
                                if ind - 1 < 0 or str(ind - 1) not in index_map.keys():
                                    track_disfluency_type('problem')
                                    track_error('problem', trans_name, utt_count)
                                else:
                                    tag+='<rm id="{}"/>'.format(index_map[str(ind - 1)])
                                tags[str(idx)] = tag
                                #adding repeted phrase to stack
                                if '>' in tokens[idx]:
                                    reparandum_stack.append(tokens[idx][:-1])
                                else:
                                    reparandum_stack.append(tokens[idx])


                            idx-=1
                        # if '<' in tokens[idx]:
                        #     tag=""
                        #     if idx > 0 and idx < len(tokens) and str(idx) in tags.keys():
                        #         tag = tags[str(idx)]
                        #     if ind - 1 < 0 or str(ind - 1) not in index_map.keys():
                        #         track_disfluency_type('problem')
                        #         track_error('problem', trans_name, utt_count)
                        #     else:
                        #         tag+='<rms id="{}"/>'.format(index_map[str(ind - 1)])
                        #     tags[str(idx)] = tag
                        #     # adding onset of repeted phrase to stack
                        #     reparandum_stack.append(tokens[idx][1:])


                        stack_size=len(reparandum_stack)
                        #print('reparandum stack')
                        #print( reparandum_stack)
                        # tagging reparandum of phrase repetition ends

                        # integrenum tagging
                        idx = ind + 1
                        prev_idx = ind - 1
                        pattern = re.compile('&[-]+[a-zA-Z]+')
                        prev_tag=False
                        if prev_idx < 0 or str(prev_idx) not in index_map.keys():
                            track_disfluency_type('problem')
                            track_error('problem', trans_name, utt_count)
                            prev_tag = True
                        while idx<len(tokens) and pattern.match(tokens[idx]) :
                            tag = ""
                            track_disfluency_type("integranum_phrase_repetition")
                            if idx < len(tokens) and str(idx) in tags.keys():
                                tag = tags[str(idx)]
                            if not prev_tag:
                                tag += '<i id="{}"/>'.format(index_map[str(prev_idx)])
                            tag += '<e/>'
                            tags[str(idx)] = tag
                            idx += 1
                        # integrenum tagging ends
                        # checking is there any fluent word in phrase rep. repair part
                        while idx < len(tokens) and tokens[idx] != reparandum_stack[-1]:
                           #
                            idx+=1
                            track_disfluency_type("fluent_repair_phrase_repetition")
                            #print('fluent word in repair part of phrase repetition')
                        # repair part of phrase repetition tagging
                        while idx < len(tokens) and len(reparandum_stack)>0 and not space.match(tokens[idx]):
                            tag=""
                            pattern = re.compile('&[-]+[a-zA-Z]+')
                            if pattern.match(tokens[idx]) :
                                track_disfluency_type("hesitation_repair_phrase_repetition")
                               # print('hesitation in repair part of phrase rep.')
                                if idx < len(tokens) and str(idx) in tags.keys():
                                    tag = tags[str(idx)]
                                tag += '<e/>'
                                tags[str(idx)] = tag


                            elif tokens[idx] not in ['[//]', '[/]', '[x'] and not space.match(tokens[idx]):
                                if len(reparandum_stack) ==stack_size:
                                    # first word of repair rep.
                                    current_word=reparandum_stack.pop()
                                  #  print('current_word :' + current_word)
                                    if(current_word==tokens[idx]):
                                        if idx < len(tokens) and str(idx) in tags.keys():
                                            tag = tags[str(idx)]
                                        if not prev_tag:
                                            tag += '<rps id="{}"/>'.format(index_map[str(prev_idx)])
                                        tags[str(idx)] = tag
                                elif len(reparandum_stack)==1:
                                    #last word of repair rep.
                                    current_word = reparandum_stack.pop()
                                   # print('current_word :'+current_word)
                                    if (current_word == tokens[idx]):
                                        if idx < len(tokens) and str(idx) in tags.keys():
                                            tag = tags[str(idx)]
                                        if not prev_tag:
                                            tag += '<rpnrep id="{}"/>'.format(index_map[str(prev_idx)])
                                        tags[str(idx)] = tag
                                else:
                                    # mid word of repair rep.
                                    current_word = reparandum_stack.pop()
                                    #print('current_word :' + current_word)
                                    if (current_word == tokens[idx]):
                                        if idx < len(tokens) and str(idx) in tags.keys():
                                            tag = tags[str(idx)]
                                        if not prev_tag:
                                            tag += '<rp id="{}"/>'.format(index_map[str(prev_idx)])
                                        tags[str(idx)] = tag
                            idx+=1
                        if len(reparandum_stack)==0:
                            print("end of repair of phrase rep. tagging")
                        else:
                            track_disfluency_type("error_repair_phrase_repetition")
                            track_error('rep._reapir_error',trans_name,utt_count)
                            #print("error in repair of phrase rep. tagging")










                    else:
                        print("word repetition")
                        print(str(tokens))

                        tag = ""
                        track_disfluency_type("word_repetition")
                        contraction_rep=False
                        #tagging reparandum of word rep.
                        if check_contraction(tokens,ind):
                            # checking word contraction
                            contraction_rep=True
                            #print('contraction in word rep. found')


                        else:
                            if ind - 1 > 0 and str(ind - 1) in tags.keys():
                                tag = tags[str(ind - 1)]
                            if ind - 1 < 0 or str(ind - 1) not in index_map.keys():
                                track_disfluency_type('problem')
                                track_error('problem', trans_name, utt_count)
                            else:
                                tag += '<rms id="{}"/>'.format(index_map[str(ind - 1)])
                            tags[str(ind - 1)] = tag  # tagging reparandum of retrace word
                            # tagging reparandum of retrace word ends
                        # check if any integranum is after the retraced word
                        idx = ind + 1
                        prev_idx = ind - 1
                        pattern = re.compile('&[-]+[a-zA-Z]+')
                        prev_tag=False
                        if prev_idx < 0 or str(prev_idx) not in index_map.keys():
                            track_disfluency_type('problem')
                            track_error('problem', trans_name, utt_count)
                            prev_tag = True
                        while idx<len(tokens) and pattern.match(tokens[idx]) :
                            tag = ""
                            if idx < len(tokens) and str(idx) in tags.keys():
                                tag = tags[str(idx)]
                            if not prev_tag:
                                tag += '<i id="{}"/>'.format(index_map[str(prev_idx)])
                            tag += '<e/>'
                            tags[str(idx)] = tag
                            idx += 1
                        # check if any integranum is after the retraced word ends

                        # tagging repair of repeted word
                        tag=""
                        if contraction_rep:
                            #print('repair word rep. in contraction cond.')
                            while idx<len(tokens) and tokens[idx]!=tokens[ind-2]:
                                #print('fluent word inside rep. repair')
                                idx+=1
                            #tagging first word of contraction repair of word rep.
                            if idx < len(tokens) and str(idx) in tags.keys():
                                tag = tags[str(idx)]
                            if ind - 1 < 0 or str(ind - 1) not in index_map.keys():
                                track_disfluency_type('problem')
                                track_error('problem', trans_name, utt_count)
                            else:
                                tag += '<rps id="{}"/>'.format(index_map[str(ind - 1)])
                            #tag += '<rpnrep id="{}"/>'.format(index_map[str(ind - 1)])
                            tags[str(idx)] = tag  # tagging repair of retrace word
                            idx+=1
                            prev_tag=False
                            if ind - 1 < 0 or str(ind - 1) not in index_map.keys():
                                track_disfluency_type('problem')
                                track_error('problem', trans_name, utt_count)
                                prev_tag = True
                            while idx<len(tokens) and tokens[idx] != tokens[ind - 1]:
                                tag=""
                                if pattern.match(tokens[idx]):
                                    track_disfluency_type('hesitation_rep._repair')

                                    if idx < len(tokens) and str(idx) in tags.keys():
                                        tag = tags[str(idx)]
                                    if not prev_tag:
                                        tag += '<i id="{}"/>'.format(index_map[str(ind-1)])
                                    tag += '<e/>'
                                    tags[str(idx)] = tag
                                else:
                                    track_disfluency_type('fluent_rep._repair')
                                    track_error('fluent_rep._repair',trans_name,utt_count)
                                    #print('fluent word inside rep. repair')
                                idx += 1
                            # tagging second word of contraction repair of word rep.

                            if idx < len(tokens) and  str(idx) in tags.keys():
                                tag = tags[str(idx)]
                            if ind - 1 < 0 or str(ind - 1) not in index_map.keys():
                                track_disfluency_type('problem')
                                track_error('problem', trans_name, utt_count)
                            else:
                                tag += '<rpnrep id="{}"/>'.format(index_map[str(ind - 1)])
                            # tag += '<rpnrep id="{}"/>'.format(index_map[str(ind - 1)])
                            tags[str(idx)] = tag  # tagging repair of retrace word

                        else:
                            while idx < len(tokens) :
                                target_word=tokens[idx]
                                if '<' in tokens[idx]:
                                    target_word=tokens[idx][1:]
                                    # tokens[idx]=tokens[idx][1:]
                                elif '>' in tokens[idx]:
                                    target_word=tokens[idx][:-1]
                                    # tokens[idx] = tokens[idx][:-1]
                                if target_word==tokens[ind-1]:
                                    break

                                idx+=1

                            if idx < len(tokens) and str(idx) in tags.keys():
                                tag = tags[str(idx)]
                            if ind - 1 < 0 or str(ind - 1) not in index_map.keys():
                                track_disfluency_type('problem')
                                track_error('problem', trans_name, utt_count)
                            else:
                                tag += '<rps id="{}"/>'.format(index_map[str(ind - 1)])
                                tag += '<rpnrep id="{}"/>'.format(index_map[str(ind - 1)])
                            tags[str(idx)] = tag  # tagging repair of retrace word

                        # tagging repair of retrace word ends
            multi_repetition = [i for i, x in enumerate(tokens) if x == "[x"]
            if len(multi_repetition) > 0:  # repetition tags
                for ind in multi_repetition:
                    pattern = re.compile('\d+]')
                    if ind > 0 and pattern.match(tokens[ind+1]):  # multi repetition
                        #print("multi repetition")
                        multi_rep_stack=[]
                        idx=ind-1
                        next_word=current_word=tokens[idx]
                        repetition_count=0
                        # pushingh multi rep. words in stack
                        while idx>=0 and current_word==next_word:

                            multi_rep_stack.append(idx)
                            repetition_count+=1

                            idx -= 1
                            if '<' in tokens[idx]:
                                next_word=tokens[idx][1:]
                            elif '>'  in tokens[idx]:
                                next_word = tokens[idx][:-1]
                            else:
                                next_word = tokens[idx]



                        # poping multi rep. words from stack and tagging incrementally


                        while len(multi_rep_stack)>0:
                            #print('multi-tag')
                            if len(multi_rep_stack)==repetition_count:



                                i=index=multi_rep_stack.pop()
                                # print(index)
                                # print(repetition_count)
                                tag = ""
                                if idx < len(tokens) and str(index) in tags.keys():
                                    tag = tags[str(index)]

                                while i<len(tokens) and str(i) in index_map.keys() and i<=index+repetition_count-2:

                                    tag += '<rms id="{}"/>'.format(index_map[str(i)])

                                    i+=1
                                tags[str(index)] = tag  # tagging first rep. word
                            elif len(multi_rep_stack)==1:
                                tag = ""
                                index = multi_rep_stack.pop()
                                if index < len(tokens) and str(index) in tags.keys():
                                    tag = tags[str(index)]
                                if index-1 < 0 or str(index-1) not in index_map.keys():
                                    track_disfluency_type('problem')
                                    track_error('problem', trans_name, utt_count)
                                else:
                                    tag += '<rps id="{}"/>'.format(index_map[str(index-1)])
                                    tag += '<rpnrep id="{}"/>'.format(index_map[str(index-1)])
                                tags[str(index)] = tag  # tagging last rep. word
                            else:
                                tag = ""
                                index = multi_rep_stack.pop()
                                if idx < len(tokens) and str(index) in tags.keys():
                                    tag = tags[str(index)]
                                if index-1 < 0 or str(index-1) not in index_map.keys():
                                    track_disfluency_type('problem')
                                    track_error('problem', trans_name, utt_count)
                                else:
                                    tag += '<rps id="{}"/>'.format(index_map[str(index - 1)])
                                    tag += '<rm id="{}"/>'.format(index_map[str(index)])
                                    tag += '<rpnrep id="{}"/>'.format(index_map[str(index - 1)])
                                tags[str(index)] = tag  # tagging mid of repeated word


            #incomplete_utt = [i for i, x in enumerate(tokens) if x == "+..."]
            #if len(incomplete_utt) > 0:  # incomplete_utt
            #    print('incomplete')
            pattern1 = re.compile('\[x')
            pattern2 = re.compile('\d+\]')


            count = 0
            index = 0
            for ind, words in enumerate(valid_words):
            # for ind,words in enumerate(tokens): # adding word, pos, tag, index per utt. in lists excluding disf. markers
                if space.match(words) :


                    continue
                if ('<' in words or '>' in words) and len(words) == 1:
                    print('checking :' + words)
                    index+=1
                    #count+=1
                    continue

                if words!='[//]' and words!='[/]' and not pattern1.match(words)and not pattern2.match(words) :

                    if '<' in words and len(words)>1:
                        words = words[1:]
                    elif '>' in words and len(words)>1:
                        words = words[:-1]
                    elif ',' in words and len(words) > 1:
                        words = words[:-1]

                    wordList.append(words)
                    if tag_error:
                        POSList.append('##') #add pos tag
                    else:
                        if count<len(tag_pos):
                            POSList.append(tag_pos[count])  # add pos tag
                        else:
                            track_disfluency_type('tag_pos list index out of range')
                            track_error('tag_pos list index out of range', trans_name, utt_count)

                    indexList.append((trans_count, utt_count, count))

                    if(str(index) in tags.keys()):
                        disfluencyTagList.append(tags[str(index)])
                    else:
                        hesitation=re.compile('&[-]+[a-zA-Z]+')
                        if hesitation.match(words):
                            disfluencyTagList.append('<e/>') # filler word tags which is not in integrenum
                        else:
                            disfluencyTagList.append('<f/>') # fluent words



                    count+=1

                index+=1
            utt_count+=1
            # print_tag(wordList, disfluencyTagList,indexList)
            uttList.append(['utt.swda_filename', str(utt_count),
                            'PAR', 'utt.damsl_act_tag()', trans_name,
                            'utt.utterance_index'])
            overallWordsList.append(wordList)  # all lists of words
            overallPOSList.append(POSList)  # all lists of corresponding POS tags
            overallTagList.append(disfluencyTagList)  # all lists of corresponding Disfluency Tags
            # all list of corresponding index tags back to the original
            # transcription
            overallIndexList.append(indexList)
            index_map.clear()
            tags.clear()
            valid_words.clear()
                        # add relevant info later about corpus

        trans_count+=1

    i=0
    print('lit_len')
    print(len(uttList))
    print(len(overallWordsList))
    corpus = ""
    easyread_corpus = ""
    for i in range(0, len(uttList)):
        uttref = uttList[i][4] + \
                 ":" + str(uttList[i][2]) + ":" \
                 + str(uttList[i][1]) + ":"+'da'


        wordstring = easy_read_disf_format(
            overallWordsList[i], overallTagList[i])
        posstring = easy_read_disf_format(
            overallPOSList[i], overallTagList[i])
        indexstring = ""
        indices = []
        for myInd in overallIndexList[i]:
            indices.append(
                "({}:{}:{})".format(myInd[0], myInd[1], myInd[2]))
        indexstring = " ".join(indices)

        # Can do easyread to the corpus string and/or more kosher xml tags
        # like the ATIS shared challenge
        easyread_format = uttref + "," + wordstring + \
                          "\nPOS," + posstring + "\n" + "REF," + indexstring
        corpus_format = detection_corpus_format(
            uttref, deepcopy(overallWordsList[i]),
            deepcopy(overallPOSList[i]),
            deepcopy(overallTagList[i]), deepcopy(indices))
        corpus += corpus_format + "\n"
        easyread_corpus += easyread_format + "\n"
    if test:
        print('after write file' + filename)
        print('count_map')
        print(count_map)
        print('count_error')
        print(count_error)
    if writeFile:
        disffile = open(target+filename + "_data.csv", "w")
        disffile.write(corpus)
        disffile.close()
        print('after write file'+ filename)
        print('count_map')
        print(count_map)
        print('count_error')
        print(count_error)




        with open('dictionary.txt', 'a') as convert_file:
            convert_file.write('after write file: '+ filename+'\n')

            convert_file.write(json.dumps(count_map))
            convert_file.write("\n")
            convert_file.write(json.dumps(count_error))
            convert_file.write("\n")
            convert_file.close()

    if writecleanFile:
        write_clean_corpus(easyread_corpus, THIS_DIR+'/../data/lm_corpora/'+filename + "_clean.text")

    if writeeditFile:
        write_edit_term_corpus(easyread_corpus, THIS_DIR+'/../data/lm_corpora/'+filename + "_edit.text")

        with open('dictionary_all.txt', 'w') as convert_file:
            convert_file.write('after write all file'+ filename+'\n')
            convert_file.write(json.dumps(count_map))
            convert_file.write("\n")
            convert_file.write(json.dumps(count_error))
            convert_file.write("\n")
            convert_file.close()
        print('after write all file'+ filename)
        print('count_map')
        print(count_map)
        print('count_error')
        print(count_error)
    count_map.clear()
    count_error.clear()





if __name__ == '__main__':
    print('hi')

    # parse command line parameters
    # Optional arguments:
    # -i string, path of source data (in swda style)
    # -t string, target path of folder for the preprocessed data
    # -f string, path of file with the division of files to be turned into
    # a corpus
    # -a string, path to disfluency annotations
    # -l string, Location of where to write a clean language\
    # model files out of this corpus
    # -pos boolean, Whether to write a word2pos mapping folder
    # in the sister directory to the corpusLocation, else assume it is there
    # -p boolean, whether to include partial words or not
    # -d boolean, include dialogue act tags in the info
    if run:
        parser = argparse.ArgumentParser(description='Feature extraction for\
           disfluency and other tagging tasks from raw data.')
        parser.add_argument('-i', action='store', dest='corpusLocation',
                            default='/../../DisfluencyProject/',
                            help='location of the corpus folder')
        parser.add_argument('-t', action='store', dest="targetDir",
                            default='/../data/disfluency_detection/DB/')
        parser.add_argument('-f', action='store', dest='divisionFile',
                            default='/../data/disfluency_detection/\
                               DB_divisions_disfluency_detection/\
                               DB_disf_train_ranges.text',
                            help='location of the file listing the \
                               files used in the corpus')

        parser.add_argument('-lm', action='store', dest='cleanModelDir',
                            default=None,
                            help='Location of where to write a clean language\
                                   model files out of this corpus.')

        args = parser.parse_args()
        corpusName = args.divisionFile[args.divisionFile.rfind("/") + 1:]. \
            replace("_ranges.text", "")
        print('corpusname :'+ corpusName)
        make_DB_corpus(True, False, False,args.targetDir, corpusName,[args.divisionFile])
    elif pos_test:
        make_DB_corpus(False, False, False, 'a', 'a', None)
    elif test:
        make_DB_corpus(False, False, False, 'a', 'a', None)

    # Example from new data with only relaxed utterances
    # available and no pos maps

    # Example from switchboard data where tree/pos maps already created
    #treeposmapdir = args.corpusLocation + '/../swda_tree_pos_maps'


