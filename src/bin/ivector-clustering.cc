#include <vector>
#include <string>
#include <math.h>
#include "util/common-utils.h"
#include "base/kaldi-common.h"
#include "ivector/plda.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/posterior.h"
#include "ilp.h"

using namespace kaldi;

void computePldaDistanceMatrix(const std::vector< Vector<double> >& vectorList, 
							Matrix<BaseFloat>& distanceMatrix,
							Plda& plda) {
	distanceMatrix.Resize(vectorList.size(),vectorList.size());
	
	// Calculate total mean and covariance:
	Vector<double> vectorMean;
	computeMean(vectorList, vectorMean);
    double scoreMean;
    double scoreVariance; 
	for (size_t i=0; i<vectorList.size();i++){
		for (size_t j=0;j<vectorList.size();j++){
			if (i == j){
				distanceMatrix(i,j) = 0;
			} else{
				distanceMatrix(i,j) = pldaScoring(vectorList[i],
													vectorList[j],
													plda);
            }
            scoreMean += distanceMatrix(i,j);
            scoreVariance += std::pow(distanceMatrix(i,j),2);
		}
	}

    int N = std::pow(vectorList.size(),2); // number of scores N = n*n
    scoreMean = scoreMean / N;
    scoreVariance = std::pow((scoreVariance / N ) - std::pow(scoreMean,2),0.5);

    for (size_t i=0; i<vectorList.size();i++){
        for (size_t j=0;j<vectorList.size();j++){
            double thisScore = distanceMatrix(i,j);
            distanceMatrix(i,j) = sigmoidRectifier((thisScore - scoreMean) / scoreVariance);
            KALDI_LOG << distanceMatrix(i,j);
        }
    }

}


int main(int argc, char *argv[]) {
    const char *usage = "Obtain glp ILP problem representation template \n";

    BaseFloat delta = 0.5;

    ParseOptions po(usage);
    po.Register("delta", &delta, "delta parameter for ILP clustering");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
        po.PrintUsage();
        exit(1);
    }

    std::string ivector_rspecifier		= po.GetArg(1),
                plda_rxfilename         = po.GetArg(2),
                ilpTemplate_wspecifier	= po.GetArg(3);

    SequentialDoubleVectorReader ivector_reader(ivector_rspecifier);

    // read plda 
    Plda plda;
    ReadKaldiObject(plda_rxfilename, &plda);
    KALDI_LOG << "plda dim: "<< plda.Dim();

	// read i-vectors
    std::vector< Vector<double> > unlabeledIvectors;
    std::vector< std::string > unlabeledIvectorIds;
    for (; !ivector_reader.Done(); ivector_reader.Next()) {
        std::string utt_label = ivector_reader.Key();
        Vector<double> utt_ivector = ivector_reader.Value();
        unlabeledIvectors.push_back(utt_ivector); 
        unlabeledIvectorIds.push_back(utt_label); 
    }
 
    KALDI_LOG << "ivector dim: " << unlabeledIvectors[0].Dim();
    // generate distant matrix from i-vectors
    Matrix<BaseFloat> distMatrix;
    computePldaDistanceMatrix(unlabeledIvectors, distMatrix, plda);

    // Generate glpk format ILP problem representation
    GlpkILP ilpObj(distMatrix, delta);
    ilpObj.glpkIlpProblem();

    // Write glpk format ILP problem template to text file
    ilpObj.Write(ilpTemplate_wspecifier);
    
    KALDI_LOG << "Wrote ILP optimization problem template to " << ilpTemplate_wspecifier;
    return 0;
}
