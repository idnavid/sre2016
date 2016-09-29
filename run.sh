#!/bin/bash

log_start(){
  echo "#####################################################################"
  echo "Spawning *** $1 *** on" `date` `hostname`
  echo ---------------------------------------------------------------------
}

log_end(){
  echo ---------------------------------------------------------------------
  echo "Done *** $1 *** on" `date` `hostname` 
  echo "#####################################################################"
}

. cmd.sh
. path.sh

set -e # exit on error

num_job=500
num_job_ubm=500
num_job_tv=40

clear_all(){
    rm -rf exp mfcc data/*/split* data/*/feats.scp data/*/cmvn.scp data/*/vad.scp data/*vad score/*
}
#clear_all

run_mfcc(){
    mfccdir=/scratch/nxs113020/mfcc
    for x in dev_JHU_subset; do
      steps/make_mfcc.sh --nj $num_job --cmd "$train_cmd" \
        data/$x exp/make_mfcc/$x $mfccdir
      steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir 
      utils/fix_data_dir_sid.sh data/$x
      #sid/compute_vad_decision.sh --nj $num_job --cmd "$train_cmd" \
      #    data/$x exp/make_vad data/${x}_vad
      utils/fix_data_dir_sid.sh data/$x
    done
}
#run_mfcc

run_unsupervised_sad(){
    sad_tools="/scratch/nxs113020/speech_activity_detection/kaldi_setup/local/"
    for x in sre16_unlabeled_minor; do
       echo "$sad_tools/compute_vad_decision.sh $num_job \"$train_cmd\" data/$x"
    done
}
#run_unsupervised_sad

ubmdim=2048
ivdim=400

run_ubm(){
  
    sid/train_diag_ubm.sh --nj $num_job_ubm --cmd "$train_cmd" data/dev ${ubmdim} \
        exp/diag_ubm_${ubmdim}
    sid/train_full_ubm.sh --nj $num_job_ubm --cmd "$train_cmd" data/dev \
        exp/diag_ubm_${ubmdim} exp/full_ubm_${ubmdim}

}
#run_ubm

run_tv_train(){

    sid/train_ivector_extractor.sh --nj $num_job_tv --cmd "$train_cmd" \
        --ivector-dim $ivdim --num-iters 5 exp/full_ubm_${ubmdim}_both_genders/final.ubm data/dev_JHU_subset \
        exp/extractor_${ubmdim}_both_genders || exit 1;

}
#run_tv_train

run_iv_extract(){

   for x in sre08; do
       sid/extract_ivectors.sh --cmd "$train_cmd" --nj $num_job \
           exp/extractor_${ubmdim}_both_genders data/$x exp/${x}.iv || exit 1;
   done

}
run_iv_extract

trials=data/trial/AIDformat_SRE10_Male_trainIndex_testIndex_keys_core_cond5.lst.kaldi
trials_key=data/trial/AIDformat_SRE10_Male_trainIndex_testIndex_keys_core_cond5.lst.kaldi.key 

run_cds_score(){
    cat $trials | awk '{print $1, $2}' | \
    ivector-compute-dot-products - \
          scp:exp/trn.iv/ivector.scp \
          scp:exp/tst.iv/ivector.scp \
          score/cds.output 2> score/cds.log
    awk '{print $3}' score/cds.output > score/cds.score
    paste score/cds.score $trials_key > score/cds.score.key           
    echo "CDS EER : `compute-eer score/cds.score.key 2> score/cds_EER`"
}
#run_cds_score

run_lda_plda(){
    mkdir -p exp/ivector_plda; rm -rf exp/ivector_plda/*

    ivector-compute-lda --dim=150 --total-covariance-factor=0.1 \
        'ark:ivector-normalize-length scp:exp/dev.iv/ivector.scp ark:- |' \
        ark:data/dev/utt2spk \
        exp/dev.iv/lda_transform.mat 2> exp/dev.iv/lda.log

    ivector-compute-plda ark:data/dev/spk2utt \
          'ark:ivector-transform exp/dev.iv/lda_transform.mat scp:exp/dev.iv/ivector.scp ark:- | ivector-normalize-length ark:-  ark:- |' \
            exp/ivector_plda/plda 2>exp/ivector_plda/plda.log
    
    ivector-plda-scoring  \
           "ivector-copy-plda --smoothing=0.0 exp/ivector_plda/plda - |" \
           "ark:ivector-transform exp/dev.iv/lda_transform.mat scp:exp/trn.iv/ivector.scp ark:- | ivector-subtract-global-mean ark:- ark:- |" \
           "ark:ivector-transform exp/dev.iv/lda_transform.mat scp:exp/tst.iv/ivector.scp ark:- | ivector-subtract-global-mean ark:- ark:- |" \
           "cat '$trials' | awk '{print \$1, \$2}' |" score/plda.output 2> score/plda.log

    awk '{print $3}' score/plda.output > score/plda.score
    paste score/plda.score $trials_key > score/plda.score.key           
    echo "PLDA EER : `compute-eer score/plda.score.key 2> score/plda_EER`"

}
#run_lda_plda



run_nda_plda() {
    mkdir -p exp/ivector_plda; rm -rf exp/ivector_plda/*

    src/bin/ivector-compute-nda \
     'ark:ivector-normalize-length scp:exp/dev.iv/ivector.scp ark:- |' \
     ark:data/dev/utt2spk \
     exp/dev.iv/nda_transform.mat 2> exp/dev.iv/nda.log

    ivector-compute-plda ark:data/dev/spk2utt \
       'ark:ivector-transform exp/dev.iv/nda_transform.mat scp:exp/dev.iv/ivector.scp ark:- | ivector-normalize-length ark:-  ark:- |' \
         exp/ivector_plda/plda 2>exp/ivector_plda/plda.log
    
    ivector-plda-scoring  \
           "ivector-copy-plda --smoothing=0.0 exp/ivector_plda/plda - |" \
           "ark:ivector-transform exp/dev.iv/nda_transform.mat scp:exp/trn.iv/ivector.scp ark:- | ivector-subtract-global-mean ark:- ark:- |" \
           "ark:ivector-transform exp/dev.iv/nda_transform.mat scp:exp/tst.iv/ivector.scp ark:- | ivector-subtract-global-mean ark:- ark:- |" \
           "cat '$trials' | awk '{print \$1, \$2}' |" score/plda.output 2> score/plda.log

    awk '{print $3}' score/plda.output > score/plda.score
    paste score/plda.score $trials_key > score/plda.score.key           
    echo "PLDA EER : `compute-eer score/plda.score.key 2> score/plda_EER`"

}
#run_nda_plda
