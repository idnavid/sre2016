// CRSS: sre2016/*/src/bin/ivector-compute-nda.cc
#include <vector>
#include <sstream>
#include <math.h>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "ivector/ivector-extractor.h"
#include "thread/kaldi-task-sequence.h"
#include <ctime>

namespace kaldi {

class NDACovarianceStats {
public:
	NDACovarianceStats(int32 dim): tot_covar_(dim),
								   between_covar_(dim),
								   num_spk_(0),
								   num_utt_(0) { }

    /// get total covariance, normalized per number of frames.
	void GetTotalCovar(SpMatrix<double> *tot_covar) const {
        	KALDI_ASSERT(num_utt_ > 0);
        	*tot_covar = tot_covar_;
        	tot_covar->Scale(1.0 / num_utt_);
	}
    void GetWithinCovar(SpMatrix<double> *within_covar) {
        KALDI_ASSERT(num_utt_ - num_spk_ > 0);
        *within_covar = tot_covar_;
        within_covar->AddSp(-1.0, between_covar_);
        within_covar->Scale(1.0 / num_utt_);
    }
    void AccStats(const Matrix<double> &utts_of_this_spk) {
        int32 num_utts = utts_of_this_spk.NumRows();
        tot_covar_.AddMat2(1.0, utts_of_this_spk, kTrans, 1.0);
        Vector<double> spk_average(Dim());
        spk_average.AddRowSumMat(1.0 / num_utts, utts_of_this_spk);
        between_covar_.AddVec2(num_utts, spk_average);
        num_utt_ += num_utts;
        num_spk_ += 1;
    }
    /// Will return Empty() if the within-class covariance matrix would be zero.
    bool SingularTotCovar() { return (num_utt_ < Dim()); }
    bool Empty() { return (num_utt_ - num_spk_ == 0); }
    std::string Info() {
        std::ostringstream ostr;
        ostr << num_spk_ << " speakers, " << num_utt_ << " utterances. ";
        return ostr.str();
    }
    int32 Dim() { return tot_covar_.NumRows(); }
    // Use default constructor and assignment operator.
    void AddStats(const NDACovarianceStats &other) {
        tot_covar_.AddSp(1.0, other.tot_covar_);
        between_covar_.AddSp(1.0, other.between_covar_);
        num_spk_ += other.num_spk_;
        num_utt_ += other.num_utt_;
    }
 private:
    KALDI_DISALLOW_COPY_AND_ASSIGN(NDACovarianceStats);
    SpMatrix<double> tot_covar_;
    SpMatrix<double> between_covar_;
    int32 num_spk_;
    int32 num_utt_;
};

template <typename T> std::string convert_to_string(T value) {
  //create an output string stream
  std::ostringstream os ;
  //throw the value into the string stream
  os << value ;
  //convert the string stream into a string and return
  return os.str() ;
}



template<class Real>
void ComputeNormalizingTransform(const SpMatrix<Real> &covar,
                                                                 MatrixBase<Real> *proj) {
    int32 dim = covar.NumRows();
    TpMatrix<Real> C(dim);    // Cholesky of covar, covar = C C^T
    C.Cholesky(covar);
    C.Invert();    // The matrix that makes covar unit is C^{-1}, because
                             // C^{-1} covar C^{-T} = C^{-1} C C^T C^{-T} = I.
    proj->CopyFromTp(C, kNoTrans);    // set "proj" to C^{-1}.
}

void ComputeAndSubtractMean(
        std::map<std::string, Vector<double> *> utt2ivector,
        Vector<double> *mean_out) {
    int32 dim = utt2ivector.begin()->second->Dim();
    size_t num_ivectors = utt2ivector.size();
    Vector<double> mean(dim);
    std::map<std::string, Vector<double> *>::iterator iter;
    for (iter = utt2ivector.begin(); iter != utt2ivector.end(); ++iter)
        mean.AddVec(1.0 / num_ivectors, *(iter->second));
    mean_out->Resize(dim);
    mean_out->CopyFromVec(mean);
    for (iter = utt2ivector.begin(); iter != utt2ivector.end(); ++iter)
        iter->second->AddVec(-1.0, *mean_out);
}

std::vector<std::pair<double, int32> > SortVector(std::vector<double>& x) {
	// Regular sort doesn't provide indexes. 
	// We sort a vector of pairs, which sorts based 
	// and rearranges based on the first element of 
	// the pairs. 
	std::vector<std::pair<double, int32> > vector_index_pair; 
	for (int i = 0; i < x.size(); i++) {
		vector_index_pair.push_back(std::make_pair(x[i],i));
	}
	std::sort(vector_index_pair.begin(),vector_index_pair.end());
	return vector_index_pair;
}


double EuclideanDistance(Vector<double>& x, Vector<double>& y) {
	Vector<double> x1(x.Dim());
	x1.CopyFromVec(x);
	Vector<double> x2(y.Dim());
	x2.CopyFromVec(y);
	x1.AddVec(-1,x2);
	return sqrt(VecVec(x1, x1));
}


double CosineDistance(const Vector<double>& v1, const Vector<double>& v2) {
	 double dotProduct = VecVec(v1, v2);
	 double norm1 = VecVec(v1, v1) + FLT_EPSILON;
	 double norm2 = VecVec(v2, v2) + FLT_EPSILON;

	 return dotProduct / (sqrt(norm1)*sqrt(norm2));  
}


void KNearestNeighbors(Vector<double> x, 
					   std::vector<Vector<double> >& Y,
					   std::map<std::string,double>& distances,
					   int32 uttIndex,
					   std::string spk_x,
					   std::string spk_y,
					   std::vector< Vector<double> >& Yout) {
	// Find and sort the k nearest neighbors from the 
	// vector x in the set of vectors Y. 
	//
	// distances was originally passed to keep record of already
	// calculated distances between i-vectors. But it ended up making
	// things slower. 
	int K = 100;
	if (K > Y.size()) {
		K = Y.size();
	}
	std::vector<double> distanceVector;
	for (int i = 0; i < Y.size(); i++) {
		double distanceValue = CosineDistance(x,Y[i]); // remove this.
		distanceVector.push_back(distanceValue);
	}
	std::vector<std::pair<double, int> > sortedDistances;
	sortedDistances = SortVector(distanceVector); 
	for (int i = 0; i < K; i++) {
		// The second element in sortedDistances contains
		// the indexes in Y.
		Yout.push_back(Y[sortedDistances[i].second]);
	}
}


double ComputeNDAWeight(Vector<double>& x,
						Vector<double>& xk,
						Vector<double>& yk){
	// The NDA weight for each sample relative to 
	// its own class and a neighboring class is the
	// ratio between the closest sample from that sample
	// to the sum of distances from the two classes. We 
	// only use the Kth nearest neighbor from each class. 
	// x: the sample at hand in class X. 
	// xk: the kth nearest neighbor from class X (self).
	// yk: the kth nearest neighbor from class Y (other).
	double dxx = CosineDistance(x, xk);
	double dxy = CosineDistance(x, yk);
	return std::min(dxx,dxy)/(dxx+dxy+1e-7);
}


Vector<double> ComputeMean(std::vector<Vector<double> > Y) {
	int K = Y.size();
	Vector<double> meanVector(Y[0].Dim());
	meanVector.SetZero();
	for (int i = 0 ; i < K; i++) {
		meanVector.AddVec(1./K,Y[i]);
	}
	return meanVector;
}


void GetVectors(std::map<std::string, std::vector<std::string> >& spk2utt,
				std::map<std::string, Vector<double> *>& utt2ivector,
				std::string& spk,
				std::vector<Vector<double> >& X) { 
	// Fetch i-vectors corresponding to speaker spk and 
	// store them in vector. 
	std::vector<std::string> utterances = spk2utt[spk];
	for (int n = 0; n < utterances.size(); n++) {
		Vector<double> ivector = *utt2ivector[utterances[n]];
		X.push_back(ivector);
	}
}



void CalculateScatterMatrices(std::map<std::string, Vector<double> *> utt2ivector, 
							   std::map<std::string, std::vector<std::string> >spk2utt, 
							   std::map<std::string,double>& distances,
							   int dim,
							   SpMatrix<double>& Sb,
							   SpMatrix<double>& Sw) {
	// Simultaneously calculates the between and within 
	// scatter matrices using NDA formulation.  
	Sb.SetZero(); // between class scatter matrix
	Sw.SetZero(); //  within class scatter matrix
	std::map<std::string, std::vector<std::string> >::iterator i;
	std::map<std::string, std::vector<std::string> >::iterator j;
	for (i = spk2utt.begin(); i!=spk2utt.end(); i++) {
		std::string spk_i = i->first;
		std::vector< Vector<double> > X_i;
		GetVectors(spk2utt,utt2ivector,spk_i, X_i);
		int N_i = X_i.size();
		for (j = spk2utt.begin(); j!=spk2utt.end(); j++) {
			std::string spk_j = j->first;
			std::vector< Vector<double> > Y_j;
			GetVectors(spk2utt,utt2ivector,spk_j,Y_j);
			if (spk_i != spk_j) {
				for (int l = 0; l < N_i; l++) {
					Vector<double> x_li = X_i[l];
					std::vector<std::string> utterances = spk2utt[spk_j];
					std::vector<Vector<double> > NNk_xy;
					KNearestNeighbors(x_li, Y_j,distances,l,spk_i,spk_j, NNk_xy);
					std::vector<Vector<double> > NNk_xx;
					KNearestNeighbors(x_li, X_i,distances,l,spk_i,spk_j,NNk_xx);
					Vector<double> M_lij = ComputeMean(NNk_xy);
					double w_lij = ComputeNDAWeight(x_li,
													NNk_xx[NNk_xx.size()-1],
													NNk_xy[NNk_xy.size()-1]);
					Vector<double> temp(dim);
					temp.SetZero();
					temp.AddVec(1,x_li);
					temp.AddVec(-1,M_lij);
					Sb.AddVec2(w_lij/(1.*N_i),temp);
				}
			}else {
				for (int l = 0; l < N_i; l++) {
					Vector<double> x_li = X_i[l];
					std::vector<std::string> utterances = spk2utt[spk_j];
					std::vector<Vector<double> > NNk_xy;
					KNearestNeighbors(x_li, Y_j,distances,l,spk_i,spk_j, NNk_xy);
					std::vector<Vector<double> > NNk_xx;
					KNearestNeighbors(x_li, X_i,distances,l,spk_i,spk_j,NNk_xx);
					Vector<double> M_lij = ComputeMean(NNk_xy);
					Vector<double> temp(dim);
					temp.SetZero();
					temp.AddVec(1,x_li);
					temp.AddVec(-1,M_lij);
					Sw.AddVec2(1./(1.*N_i),temp);
				}
			}
		}
	}
}


void ComputeNDATransform(const std::map<std::string, Vector<double> *> &utt2ivector,
						 const std::map<std::string, std::vector<std::string> > &spk2utt,
						 double total_covariance_factor,
						 MatrixBase<double> *nda_out) {
	KALDI_ASSERT(!utt2ivector.empty());
	int32 nda_dim = nda_out->NumRows(), dim = nda_out->NumCols();
	KALDI_ASSERT(dim == utt2ivector.begin()->second->Dim());
	KALDI_ASSERT(nda_dim > 0 && nda_dim <= dim);

	NDACovarianceStats stats(dim);
    
	std::map<std::string, std::vector<std::string> >::const_iterator iter;
	for (iter = spk2utt.begin(); iter != spk2utt.end(); ++iter) {
		const std::vector<std::string> &uttlist = iter->second;
		KALDI_ASSERT(!uttlist.empty());

		int32 N = uttlist.size(); // number of utterances for spkr
		Matrix<double> utts_of_this_spk(N, dim);
		for (int32 n = 0; n < N; n++) {
			std::string utt = uttlist[n];
			KALDI_ASSERT(utt2ivector.count(utt) != 0);
			utts_of_this_spk.Row(n).CopyFromVec(*(utt2ivector.find(utt)->second));
		}
		stats.AccStats(utts_of_this_spk);
	}
	KALDI_LOG << "Stats have " << stats.Info();
	KALDI_ASSERT(!stats.Empty());
	KALDI_ASSERT(!stats.SingularTotCovar() &&
						"Too little data for iVector dimension.");


	SpMatrix<double> total_covar;
	stats.GetTotalCovar(&total_covar);
	std::map<std::string,double> distances;
	SpMatrix<double> between_covar(dim);
	SpMatrix<double> within_covar(dim);
	CalculateScatterMatrices(utt2ivector,
							 spk2utt,
							 distances,
							 dim,
							 between_covar,
							 within_covar);
	
	SpMatrix<double> mat_to_normalize(dim);
	mat_to_normalize.AddSp(total_covariance_factor, total_covar);
	mat_to_normalize.AddSp(1.0 - total_covariance_factor, within_covar);
	
	Matrix<double> T(dim, dim); 
	ComputeNormalizingTransform(mat_to_normalize, &T);
	SpMatrix<double> between_covar_proj(dim);
	between_covar_proj.AddMat2Sp(1.0, T, kNoTrans, between_covar, 0.0);

	Matrix<double> U(dim, dim);
	Vector<double> s(dim);
	between_covar_proj.Eig(&s, &U);
	bool sort_on_absolute_value = false; // any negative ones will go last (they
										 // shouldn't exist anyway so doesn't
                                            // really matter)
	SortSvd(&s, &U, static_cast<Matrix<double>*>(NULL),
			sort_on_absolute_value);
	
	KALDI_LOG << "Singular values of between-class covariance after projecting "
			  << "with interpolated [total/within] covariance with a weight of "
			  << total_covariance_factor << " on the total covariance, are: " << s;

	// U^T is the transform that will diagonalize the between-class covariance.
	// U_part is just the part of U that corresponds to the kept dimensions.
	SubMatrix<double> U_part(U, 0, dim, 0, nda_dim);

	// We first transform by T and then by U_part^T.    This means T
	// goes on the right.
	Matrix<double> temp(nda_dim, dim);
	temp.AddMatMat(1.0, U_part, kTrans, T, kNoTrans, 0.0);
	nda_out->CopyFromMat(temp);
}

}

int main(int argc, char *argv[]) {
	using namespace kaldi;
	typedef kaldi::int32 int32;

	const char *usage = 
		"ivector-compute-nda [options] ark:ivectors.ark ark:utt2spk nda.mat\n";
	ParseOptions po(usage);
	po.Read(argc,argv);
	int32 nda_dim = 150;
	double total_covariance_factor = 0.1;
	bool binary = true;        

	if (po.NumArgs() != 3) {
		po.PrintUsage();
		exit(1);
	}
	std::string ivector_rspecifier = po.GetArg(1),
				utt2spk_rspecifier = po.GetArg(2),
				nda_wxfilename = po.GetArg(3);

	SequentialBaseFloatVectorReader ivector_reader(ivector_rspecifier);
	RandomAccessTokenReader utt2spk_reader(utt2spk_rspecifier);
	
	std::map<std::string, Vector<double> *> utt2ivector;
	std::map<std::string, std::vector<std::string> > spk2utt;

	int32 num_err = 0;
	int32 num_done = 0;
	int32 dim = 0;
	for (; !ivector_reader.Done(); ivector_reader.Next()) {
		std::string utt = ivector_reader.Key();
		const Vector<BaseFloat> &ivector = ivector_reader.Value();
		if (utt2ivector.count(utt) != 0) {
			KALDI_WARN << "Duplicate iVector found for utterance " << utt
					     << ", ignoring it.";
			num_err++;
			continue;
		}
		if (!utt2spk_reader.HasKey(utt)) {
			KALDI_WARN << "utt2spk has no entry for utterance " << utt
					     << ", skipping it.";
			num_err++;
			continue;
		}
		std::string spk = utt2spk_reader.Value(utt);
		utt2ivector[utt] = new Vector<double>(ivector);
		if (dim == 0) {
			dim = ivector.Dim();
		} else {
			KALDI_ASSERT(dim == ivector.Dim() && "iVector dimension mismatch");
		}
		spk2utt[spk].push_back(utt);
		num_done++;
	}

	KALDI_LOG << "Read " << num_done << " utterances, "
			    << num_err << " with errors.";

	if (num_done == 0) {
		KALDI_ERR << "Did not read any utterances.";
	} else {
		KALDI_LOG << "Computing within-class covariance.";
	}

	Vector<double> mean;
	ComputeAndSubtractMean(utt2ivector, &mean);
	KALDI_LOG << "2-norm of iVector mean is " << mean.Norm(2.0);

	Matrix<double> nda_mat(nda_dim, dim + 1); 
	SubMatrix<double> linear_part(nda_mat, 0, nda_dim, 0, dim);
	ComputeNDATransform(utt2ivector,
						spk2utt,
						total_covariance_factor,
						&linear_part);
	Vector<double> offset(nda_dim);
	offset.AddMatVec(-1.0, linear_part, kNoTrans, mean, 0.0);
    nda_mat.CopyColFromVec(offset, dim); // add mean-offset to transform
	
	KALDI_VLOG(2) << "2-norm of transformed iVector mean is "
				    << offset.Norm(2.0);
	
	WriteKaldiObject(nda_mat, nda_wxfilename, binary);

	KALDI_LOG << "Wrote LDA transform to "
			    << PrintableWxfilename(nda_wxfilename);
	
	std::map<std::string, Vector<double> *>::iterator iter;
	for (iter = utt2ivector.begin(); iter != utt2ivector.end(); ++iter)
		delete iter->second;
	utt2ivector.clear();

	return 0;
}