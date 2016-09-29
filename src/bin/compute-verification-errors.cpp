//
//  compute-verification-errors.cpp
//  
//
//  Created by Navid Shokouhi on 9/28/16.
//
//

#include "base/kaldi-common.h"
#include "util/common-utils.h"

namespace kaldi {
    
    /**
     ComputeEer computes the Equal Error Rate (EER) for the given scores
     and returns it as a proportion beween 0 and 1.
     If we set the threshold at x, then the target error-rate is the
     proportion of target_scores below x; and the non-target error-rate
     is the proportion of non-target scores above x.  We seek a
     threshold x for which these error rates are the same; this
     error rate is the EER.
     
     We compute this by iterating over the positions in target_scores: 0, 1, 2,
     and so on, and for each position consider whether the cutoff could be here.
     For each of these position we compute the corresponding position in
     nontarget_scores where the cutoff would be if the EER were the same.
     For instance, if the vectors had the same length, this would be position
     length() - 1, length() - 2, and so on.  As soon as the value at that
     position in nontarget_scores at that position is less than the value from
     target_scores, we have our EER.
     
     In coding this we weren't particularly careful about edge cases or
     making sure whether it's actually n + 1 instead of n.
     */
    
    std::vector<BaseFloat> ComputeErrors(std::vector<BaseFloat> *target_scores,
                                         std::vector<BaseFloat> *nontarget_scores,
                                         BaseFloat *threshold,
                                         BaseFloat Cmiss,
                                         BaseFloat Cfa,
                                         BaseFloat Ptarget) {
        KALDI_ASSERT(!target_scores->empty() && !nontarget_scores->empty());
        std::sort(target_scores->begin(), target_scores->end());
        std::sort(nontarget_scores->begin(), nontarget_scores->end());
        
        BaseFloat eer;
        size_t target_position = 0,
        target_size = target_scores->size();
        for (; target_position + 1 < target_size; target_position++) {
            // Pmiss = target_position / target_size
            // define number_of_nontargets_beyond_thr as
            // In that case,
            // if (Pmiss == Pfa) number_of_nontargets_beyond_thr = Pmiss * num_of_nontarget
            //      since number_of_nontargets_beyond_thr = Pfa * num_of_nontarget
            ssize_t nontarget_size = nontarget_scores->size(),
            nontarget_n = nontarget_size * target_position * 1.0 / target_size,
            nontarget_position = nontarget_size - 1 - nontarget_n;
            
            if (nontarget_position  < 0) {
                nontarget_position = 0;
            }
            if ((*nontarget_scores)[nontarget_position] <
                (*target_scores)[target_position]) {
                *threshold = (*target_scores)[target_position];
                eer = target_position * 1.0 / target_size;
                break;
            }
        }


        BaseFloat minDcf = 1000.;
        BaseFloat dcf = 1000.;
        target_position = 0;
        for (; target_position + 1 < target_size; target_position++) {
            ssize_t nontarget_size = nontarget_scores->size();
            
            // number of misses for this threshold:
            BaseFloat Pmiss = target_position * 1.0 / target_size;
            BaseFloat thr = (*target_scores)[target_position];
            BaseFloat Pfa = 0.;
            for (int i = nontarget_size - 1; i >= 0; i--) {
                if (thr < (*nontarget_scores)[i]) {
                    Pfa++;
                } else { 
                    break;
                }
            }
            Pfa = Pfa * 1. / nontarget_size;
            dcf = Cmiss * Pmiss * Ptarget + Cfa * Pfa * ( 1 - Ptarget);
            if (dcf < minDcf) {
                minDcf = dcf;  
            }
            
        }
         
            

        std::vector<BaseFloat> errors;
        errors.push_back(eer);
        errors.push_back(100*minDcf);
        return errors;
    }
    
}



int main(int argc, char *argv[]) {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    try {
        const char *usage =
        "Computes Equal Error Rate and minimum detection cost function\n"
        "Input is a series of lines, each with two fields.\n"
        "The first field must be a numeric score, and the second\n"
        "either the string 'target' or 'nontarget'. \n"
        "The EER will be printed to the standard output.\n"
        "\n"
        "Usage: compute-verification-errors <scores-in> miss-cost falsealarm-cost target-prior\n"
        "E.g. compute-verification-errors score/plda.score.key 10 1 0.01";
        
        ParseOptions po(usage);
        po.Read(argc, argv);
        
        if (po.NumArgs() != 4) {
            po.PrintUsage();
            exit(1);
        }
        
        std::string scores_rxfilename = po.GetArg(1),
                    Cmiss_string      = po.GetArg(2),
                    Cfa_string        = po.GetArg(3),
                    ptarget_string    = po.GetArg(4);
        

        BaseFloat Cmiss     = std::atof(Cmiss_string.c_str());
        BaseFloat Cfa       = std::atof(Cfa_string.c_str());
        BaseFloat Ptarget   = std::atof(ptarget_string.c_str());

        std::vector<BaseFloat> target_scores, nontarget_scores;
        Input ki(scores_rxfilename);
        
        std::string line;
        while (std::getline(ki.Stream(), line)) {
            std::vector<std::string> split_line;
            SplitStringToVector(line, " \t", true, &split_line);
            BaseFloat score;
            if (split_line.size() != 2) {
                KALDI_ERR << "Invalid input line (must have two fields): "
                << line;
            }
            if (!ConvertStringToReal(split_line[0], &score)) {
                KALDI_ERR << "Invalid input line (first field must be float): "
                << line;
            }
            if (split_line[1] == "target")
                target_scores.push_back(score);
            else if (split_line[1] == "nontarget")
                nontarget_scores.push_back(score);
            else {
                KALDI_ERR << "Invalid input line (second field must be "
                << "'target' or 'nontarget')";
            }
        }
        if (target_scores.empty() && nontarget_scores.empty())
            KALDI_ERR << "Empty input.";
        if (target_scores.empty())
            KALDI_ERR << "No target scores seen.";
        if (nontarget_scores.empty())
            KALDI_ERR << "No non-target scores seen.";
        
        BaseFloat threshold;
        std::vector<BaseFloat> errors = ComputeErrors(&target_scores, &nontarget_scores, &threshold,
                                      Cmiss, Cfa, Ptarget);
        BaseFloat eer = errors[0];
        BaseFloat minDcf = errors[1];
        KALDI_LOG << "Equal error rate is " << (100.0 * eer)
        << "%, at threshold " << threshold;
        KALDI_LOG << "minDCF is " << minDcf;
        
        std::cout.precision(4);
        std::cout << (100.0 * eer);
        std::cout << (minDcf);
        
        return 0;
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}