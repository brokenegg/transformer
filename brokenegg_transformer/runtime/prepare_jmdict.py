# Copyright 2020 Katsuya Iida. All rights reserved.

'''Parse XML file defined by http://www.edrdg.org/jmdict/jmdict_dtd_h.html'''

import argparse
import xml.sax
import gzip
import re
import os
from collections import namedtuple
from typing import List

JMDictKEle = namedtuple('JMDictKEle', ('keb', 'ke_inf_list', 'ke_pri_list'))
JMDictREle = namedtuple('JMDictREle', ('reb', 're_inf_list', 're_pri_list'))
JMDictSense = namedtuple('JMDictSense', ('stagk_list', 'stagr_list', 'gloss_list'))
JMDictGloss = namedtuple('JMDictGloss', ('lang', 'text'))

MAX_GLOSSES = 3

lang_map = {
    'eng': 'en',
    'dut': 'nl',
    'fre': 'fr',
    'ger': 'de',
    'hun': 'hun',
    'rus': 'ru',
    'slv': 'sl',
    'spa': 'es',
    'swe': 'sv',
}

def reduce_k_ele_list(k_ele_list) -> List[str]:
    return [
        k_ele.keb
        for k_ele in k_ele_list
        if not k_ele.ke_inf_list and k_ele.ke_pri_list
    ]

def reduce_r_ele_list(r_ele_list):
    return [
        r_ele.reb
        for r_ele in r_ele_list
        if not r_ele.re_inf_list and r_ele.re_pri_list
    ]

GLOSS_TEXT_RX1 = re.compile(r'\(.*\)')
GLOSS_TEXT_RX2 = re.compile(r'^.*\)')
GLOSS_TEXT_RX3 = re.compile(r'\(.*$')
GLOSS_TEXT_RX4 = re.compile(r'{.*}')
GLOSS_TEXT_RX5 = re.compile(r'^.*}')
GLOSS_TEXT_RX6 = re.compile(r'{.*$')
GLOSS_TEXT_RX7 = re.compile(r';.*$')
GLOSS_TEXT_RX8 = re.compile(r'[ぁ-んァ-ンー\u4e00-\u9fff]')

def reduce_gloss_text(text):
    text = GLOSS_TEXT_RX1.sub('', text)
    text = GLOSS_TEXT_RX2.sub('', text)
    text = GLOSS_TEXT_RX3.sub('', text)
    text = GLOSS_TEXT_RX4.sub('', text)
    text = GLOSS_TEXT_RX5.sub('', text)
    text = GLOSS_TEXT_RX6.sub('', text)
    text = text.strip()
    text = text.replace(':', '').strip()
    if GLOSS_TEXT_RX8.search(text):
        return ''
    return text

def reduce_gloss_list(eb, gloss_list):
    res = []
    for gloss in gloss_list:
        text = reduce_gloss_text(gloss.text)
        lang = lang_map[gloss.lang]
        if lang == 'ru':
            text = GLOSS_TEXT_RX7.sub('', text).strip()
            if text:
                for text2 in text.split(','):
                    text2 = text2.strip()
                    if text2:
                        if len(eb) > len(text2.split()):
                            res.append((lang, eb, text2))
        else:
            if text:
                if len(eb) > len(text.split()):
                    res.append((lang, eb, text))
    return res

def reduce_sense(keb_list: List[str], reb_list: List[str], sense):
    res = []
    if sense.stagk_list or sense.stagr_list:
        if keb_list:
            for stagk in sense.stagk_list:
                if stagk in keb_list:
                    t = reduce_gloss_list(stagk, sense.gloss_list)
                    res.extend(t)
        if reb_list:
            for stagr in sense.stagr_list:
                if stagr in reb_list:
                    t = reduce_gloss_list(stagr, sense.gloss_list)
                    res.extend(t)
    else:
        if keb_list:
            for keb in keb_list:
                t = reduce_gloss_list(keb, sense.gloss_list)
                res.extend(t)
        if reb_list:
            for reb in reb_list:
                t = reduce_gloss_list(reb, sense.gloss_list)
                res.extend(t)
    return res


class JMDictXMLHandler(xml.sax.ContentHandler):
    def __init__(self, elem_fn):
        self.tags = []
        self.text = ''
        self.elem_fn = elem_fn
        self.keb = None

    def startElement(self, tag, attributes):
        if tag == 'entry':
            self.k_ele_list = []
            self.r_ele_list = []
            self.sense_list = []
        elif tag == 'k_ele':
            self.keb = None
            self.ke_inf_list = []
            self.ke_pri_list = []
        elif tag == 'keb':
            pass
        elif tag == 'ke_inf':
            pass
        elif tag == 'ke_pri':
            pass
        elif tag == 'r_ele':
            self.reb = None
            self.re_inf_list = []
            self.re_pri_list = []
        elif tag == 'reb':
            pass
        elif tag == 're_inf':
            pass
        elif tag == 're_pri':
            pass
        elif tag == 'sense':
            self.gloss_list = []
            self.stagk_list = []
            self.stagr_list = []
        elif tag == 'gloss':
            self.lang = attributes.get('xml:lang')
        self.tags.append(tag)
        self.text = ''

    def endElement(self, tag):
        text = self.text
        text = re.sub('\s+', ' ', text).strip()
        if text:
            pass #print('/'.join(self.tags) + ':' + text)
        assert tag == self.tags.pop()
        if tag == 'JMdict':
            pass
        elif tag == 'entry':
            self.elem_fn(self.k_ele_list, self.r_ele_list, self.sense_list)
        elif tag == 'ent_seq':
            pass
        elif tag == 'k_ele':
            self.k_ele_list.append(JMDictKEle(self.keb, self.ke_inf_list, self.ke_pri_list))
        elif tag == 'keb':
            self.keb = text
            assert self.keb
        elif tag == 'ke_inf':
            self.ke_inf_list.append(text)
        elif tag == 'ke_pri':
            self.ke_pri_list.append(text)
        elif tag == 'r_ele':
            self.r_ele_list.append(JMDictREle(self.reb, self.re_inf_list, self.re_pri_list))
        elif tag == 'reb':
            self.reb = text
            assert self.reb
        elif tag == 're_nokanji':
            pass #print('re_nokanji', text)
        elif tag == 're_restr':
            pass #print('re_restr', text)
        elif tag == 're_inf':
            self.re_inf_list.append(text)
        elif tag == 're_pri':
            self.re_pri_list.append(text)
        elif tag == 'sense':
            self.sense_list.append(JMDictSense(self.stagk_list, self.stagr_list, self.gloss_list))
            pass
        elif tag == 'stagk':
            self.stagk_list.append(text)
        elif tag == 'stagr':
            self.stagr_list.append(text)
        elif tag == 'pos':
            pass
        elif tag == 'xref':
            pass
        elif tag == 'ant':
            pass
        elif tag == 'field':
            pass
        elif tag == 'misc':
            pass
        elif tag == 's_inf':
            pass
        elif tag == 'lsource':
            pass
        elif tag == 'dial':
            pass
        elif tag == 'gloss':
            self.gloss_list.append(JMDictGloss(self.lang, text))
        elif tag == 'pri':
            print('xxxx pri', text)
        else:
            raise ValueError(f"Unknown tag {tag}")
        self.text = ''

    def characters(self, characters):
        self.text += characters

def elem_fn(k_ele_list, r_ele_list, sense_list):
    ele_list, sense_list = reduce_entry(k_ele_list, r_ele_list, sense_list)
    if ele_list:
        print('---')
        for k_ele in k_ele_list:
            print(k_ele)
        for r_ele in r_ele_list:
            print(r_ele)
        for gloss_list in sense_list:
            print('  ---')
            for gloss in gloss_list:
                print('  ', gloss)

def parse_jmdict(input_file, output_dir):

    f1 = {
        lang: gzip.open(os.path.join(output_dir, 'jmdict.ja-%s.txt.gz' % lang), 'wt')
        for lang in lang_map.values()
    }
    f2 = {
        lang: gzip.open(os.path.join(output_dir, 'jmdict.%s-ja.txt.gz' % lang), 'wt')
        for lang in lang_map.values()
    }

    def elem_fn2(k_ele_list, r_ele_list, sense_list):
        keb_list = reduce_k_ele_list(k_ele_list)
        reb_list = reduce_r_ele_list(r_ele_list)
        if keb_list or reb_list:
            for sense in sense_list:
                para_list = reduce_sense(keb_list, reb_list, sense)
                for lang, eb, text in para_list[:MAX_GLOSSES]:
                    f1[lang].write('%s\t%s\n' % (eb, text))
                for lang, eb, text in para_list:
                    f2[lang].write('%s\t%s\n' % (text, eb))

    parser = xml.sax.make_parser()
    handler = JMDictXMLHandler(elem_fn2)
    parser.setContentHandler(handler)
    with gzip.open(input_file, 'rt') as f:
        parser.parse(f)

    for f in f1.values(): f.close()
    for f in f2.values(): f.close()

def make_tfrecord(raw_dir, data_dir, vocab_file):
    from brokenegg_transformer.data_download import (
        encode_and_save_files, get_vocab_file, shuffle_records)
    subtokenizer = get_vocab_file(raw_dir, data_dir, vocab_file)
    langs = 'de', 'en', 'es', 'fr', 'ru'
    for lang in langs:
        for lang_pair in '%s-ja' % lang, 'ja-%s' % lang:
            raw_file = os.path.join(raw_dir, 'jmdict.%s.txt.gz' % lang_pair)
            train_tfrecord_files, _ = encode_and_save_files(subtokenizer, data_dir, lang_pair, [raw_file],
                2, 0, 0.0, prefix='jmdict', input_column=0, target_column=1, randomize_input=0.0)
            for fname in train_tfrecord_files:
                shuffle_records(fname)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--vocab')
    parser.add_argument('--mode')
    args = parser.parse_args()
    if args.mode == 'split':
        parse_jmdict(args.input, args.output)
    elif args.mode == 'tfrecord':
        make_tfrecord(args.input, args.output, args.vocab)
    else:
        raise ValueError()

if __name__ == '__main__':
    main()