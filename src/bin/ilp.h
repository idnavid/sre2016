#ifndef KALDI_IVECTOR_DIAR_ILP_H_
#define KALDI_IVECTOR_DIAR_ILP_H_

#include <vector>
#include <string>
#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/posterior.h"
#include "/scratch2/nxs113020/CRSSdiar/src/diar/diar-utils.h"

namespace kaldi{

// The ILP clustering approach implemented in this file uses refers to the paper
// [1] "Recent Improvements on ILP-based Clustering for Broadcast news speaker diarization",
// by Gregor Dupuy, Sylvain Meignier, Paul Deleglise, Yannic Esteve

typedef kaldi::int32 int32;

class GlpkILP {
public:
	GlpkILP() {};
	GlpkILP(BaseFloat delta);
	GlpkILP(Matrix<BaseFloat>& distanceMatrix, BaseFloat delta);

	// generate ILP problem description in CPLEX LP format
	void glpkIlpProblem();

	// write objective function of ILP in glpk format, refer to equation (2) in the paper [1]
	std::string problemMinimize();

	// write constraint function for unique center assigment as in equation (2.3) in the paper[1]
	void problemConstraintsColumnSum();

	//  write constraint function as in equation (2.4) in the paper[1]
	void problemConstraintsCenter();

	// explicitly enforce distance upperbound (eq. 1.5) in paper [1]
	void distanceUpperBound();

	// list all binary variables as in equation (2.2) in the paper [1]
	void listBinaryVariables();


	// generate variable names represent ILP problem in glpk format
	std::string indexToVarName(std::string variableName, int32 i, int32 j);

	// generate variable names represent ILP problem in glpk format
	std::vector<int32> varNameToIndex(std::string& variableName);

	// write template into filse
	void Write(std::string outName);

	// Read the ILP solution (written in glpk format)
	std::vector<std::string> ReadGlpkSolution(std::string glpkSolutionFile);

private:
	BaseFloat _delta;
	std::vector<std::string> _problem;
	Matrix<BaseFloat> _distanceMatrix;
};

}


#endif 