// Machine Learning Winter School, Inha University (Feb 5~7, 2024)
// written by Changhwan Choi
// Db score distribution for each flavor jet (ATL-PHYS-PUB-2022-027)

#include <iostream>
#include <vector>

#include "TMath.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TGraph2D.h"
#include "TGraph.h"
#include "TLine.h"
#include "TCanvas.h"
#include "TLegend.h"


// D_b histogram
#define DB_NBINS 100u
#define DB_FROM -10.f
#define DB_TO 35.f
#define DB_Y_FROM 0.5e-3f
#define DB_Y_TO 10000.e-2f

#define WEIGHT true

// Thresholds
#define NTHRES 1000u
#define THRES_FROM -5.f
#define THRES_TO 30.f

// Parameter f_c: used for D_b calculation. Weight parameter between c and l.
#define PARAM_F 0.018f

// Jet p_T
#define PT_NBINS 12u
const float pt_bins[PT_NBINS+1] = {10.f, 15.f, 20.f, 25.f, 30.f, 35.f, 40.f, 45.f, 50.f, 60.f, 70.f, 85.f, 100.f};
const float temp_ratio[3] = {1.f, 2.14f, 27.0f}; // b : c : l for jet pT 50~60GeV/c, from analysis note
#define PT_LB 50.f
#define PT_UB 60.f


// SV performances for jet pT 50~60GeV/c, from analysis note
#define SV_NTHRES 14u
const float sv_eff[SV_NTHRES] = {0.2517, 0.3406, 0.3872, 0.2367, 0.3170, 0.3586, 0.3785, 0.2208, 0.2979, 0.3390, 0.2134, 0.2802, 0.3167, 0.3315};
const float sv_fpr[SV_NTHRES] = {0.0307, 0.0513, 0.0700, 0.0243, 0.0408, 0.0556, 0.0699, 0.0200, 0.0338, 0.0467, 0.0177, 0.0288, 0.0398, 0.0508};
const float sv_pur[SV_NTHRES] = {0.4222, 0.3718, 0.3303, 0.4650, 0.4094, 0.3650, 0.3256, 0.4957, 0.4402, 0.3927, 0.5179, 0.4641, 0.4150, 0.3675};


// Enum & consts
enum flav {b, c, l};
const char flav_c[3] = {'b', 'c', 'l'};
const int flav_color[4] = {kAzure-4, kPink-4, kOrange-4, 15};


// Calculates D_b score with softmax scores (float[3])
inline float Db(const float* p) {
    if (p[c] == 0.f && p[l] == 0.f)
        assert(TString::Format("ERROR: p = (%f, %f, %f) -> divide by zero", p[b], p[c], p[l]));
    return TMath::Log( p[b] / ((1.f - PARAM_F) * p[l] + PARAM_F * p[c]) );
}

// Gets pT bin index
int GetPtBin(float pt) {
    int bin;
    for (bin=0; bin<PT_NBINS+1; bin++) {
        if (pt < pt_bins[bin])
            return bin;
    }
    return bin; // return 0 if pT is lower than 10GeV/c, PT_NBINS+1 if pT is higher than 100GeV/c
}


// Draw D_b distribution and ROC curve along D_b thresholds.
void mlschool_eval_jc(void) {
    // Load .root file
    TFile* tf = new TFile("eval_jc_reduc.root", "READ");
    TTree* tr = (TTree*)tf->Get("tree");
    
    Long64_t jet_flav;
    float jet_pt;
    float softmax[3];

    tr->SetBranchAddress("jet_flav", &jet_flav);
    tr->SetBranchAddress("jet_pt", &jet_pt);
    for (int f=0; f<3; f++)
        tr->SetBranchAddress(TString::Format("softmax-%c", flav_c[f]), &softmax[f]);
    

    // Init result arrays
    float pred_result[NTHRES][3][2] = {0.f}; // [thres_idx][truth_flav][pred_bool], to take b:c:l ratio into account
    
    float tpr[NTHRES+2], fpr[NTHRES+2];
    float eff[NTHRES], pur[NTHRES], thres_arr[NTHRES];


    // Init histograms for calculating pT range scale factor
    TH1F* hist_pt_truth_flav_data[3];
    for (int f=b; f<=l; f++)
        hist_pt_truth_flav_data[f] = (TH1F*)tf->Get(TString::Format("hist_pt_label-%c", flav_c[f]));
    TH1F* hist_pt_truth_flav_real[3];
    // temp (1:2:25)
    for (int f=b; f<=l; f++) {
        hist_pt_truth_flav_real[f] = new TH1F(TString::Format("hist_pt_label-%c_real", flav_c[f]), "", PT_NBINS, pt_bins);
        for (int bin=0; bin<PT_NBINS; bin++)
            hist_pt_truth_flav_real[f]->Fill(pt_bins[bin], temp_ratio[f]);
    } // ~temp

    TH1F hist_pt_incl_real = *hist_pt_truth_flav_real[b];
    hist_pt_incl_real.SetName("hist_pt_incl_real");
    hist_pt_incl_real.Add(hist_pt_truth_flav_real[c]);
    hist_pt_incl_real.Add(hist_pt_truth_flav_real[l]);

    // pT range scale factor = (relative num of x-jet, incl) / (num of x-jet, data) / (sum of relative num of x-jet, incl)
    // -> eff/pur is for mean value along jet pT bins
    // -> standard to select WP for entire pT ranges
    TH1F hist_pt_scale[3];
    for (int f=b; f<=l; f++) {
        hist_pt_scale[f] = *hist_pt_truth_flav_real[f];
        hist_pt_scale[f].SetName(TString::Format("hist_pt_scale-%c", flav_c[f]));
        hist_pt_scale[f].Divide(hist_pt_truth_flav_data[f]);
        hist_pt_scale[f].Divide(&hist_pt_incl_real);
    }


    // Init hist Db
    TH1F* hist_db[3];
    for (int f=b; f<=l; f++) {
        hist_db[f] = new TH1F(TString::Format("hist_db_%d", f), "D_{b}", DB_NBINS, DB_FROM, DB_TO);
        hist_db[f]->SetLineColor(flav_color[f]);
        hist_db[f]->SetLineWidth(2);
        hist_db[f]->GetXaxis()->SetTitle("D_{b}");
        hist_db[f]->GetYaxis()->SetTitle("a.u.");
        hist_db[f]->GetYaxis()->SetRangeUser(DB_Y_FROM, DB_Y_TO);
        hist_db[f]->SetStats(false);
    }
    

    // Fill hist Db and result array values along thresholds
    Long64_t entries = tr->GetEntries();
    for (int e=0; e<entries; e++) {
        tr->GetEntry(e);
        if (jet_pt >= PT_UB || jet_pt < PT_LB)
            continue;

        float scale_factor = WEIGHT ? hist_pt_scale[jet_flav].GetBinContent(GetPtBin(jet_pt)) : 1.f;

        float db = Db(softmax);
        hist_db[jet_flav]->Fill(db, scale_factor);
        for (int t=0; t<NTHRES; t++) {
            float thres = (THRES_TO - THRES_FROM) * float(t)/float(NTHRES) + THRES_FROM;
            pred_result[t][jet_flav][db > thres] += scale_factor;
            // thres_arr[t] = thres;
        }
    }

    // DEBUG: Export hist scale factor
    // TCanvas* canvas_scale_factor = new TCanvas();
    // gPad->SetLogy();
    // for (int f=b; f<=l; f++) {
    //     hist_pt_scale[f].SetLineColor(flav_color[f]);
    //     hist_pt_scale[f].GetYaxis()->SetRangeUser(0.00001, 0.1);
    //     hist_pt_scale[f].SetStats(false);
    //     hist_pt_scale[f].Draw("HIST;SAME");
    // }
    // canvas_scale_factor->Print("images/scale_factor.pdf");


    // Calc TPR, FPR for ROC curve, and Eff, Pur along thresholds
    for (int t=0; t<NTHRES; t++) {
        float tp = pred_result[t][b][true];
        float fp = pred_result[t][c][true] + pred_result[t][l][true];
        float tn = pred_result[t][c][false] + pred_result[t][l][false];
        float fn = pred_result[t][b][false];
        
        tpr[t+1] = tp / (tp + fn);
        fpr[t+1] = fp / (fp + tn);

        eff[t] = tp / (tp + fn);
        pur[t] = tp / (tp + fp);
    }
    tpr[0] = 1.f; tpr[NTHRES+1] = 0.f;
    fpr[0] = 1.f; fpr[NTHRES+1] = 0.f;


    // Init ROC curve
    TGraph* roc_curve = new TGraph(NTHRES+2, fpr, tpr);
    roc_curve->SetName("roc_curve");
    roc_curve->SetTitle("ROC curve");
    roc_curve->GetXaxis()->SetTitle("FP Rate");
    roc_curve->GetYaxis()->SetTitle("TP Rate");
    roc_curve->GetXaxis()->SetRangeUser(0.f, 1.f);
    roc_curve->GetYaxis()->SetRangeUser(0.f, 1.f);
    roc_curve->SetLineColor(kRed);
    roc_curve->SetLineWidth(2);


    // Init Eff vs. Pur
    // TGraph2D* eff_vs_pur = new TGraph2D(NTHRES, eff, pur, thres_arr);
    TGraph* eff_vs_pur = new TGraph(NTHRES, eff, pur);
    eff_vs_pur->SetName("eff_vs_pur");
    eff_vs_pur->SetTitle("");
    eff_vs_pur->GetXaxis()->SetTitle("Efficiency");
    eff_vs_pur->GetYaxis()->SetTitle("Purity");
    // eff_vs_pur->GetZaxis()->SetTitle("D_{b}^{min}");
    eff_vs_pur->GetXaxis()->SetRangeUser(0.f, 1.f);
    eff_vs_pur->GetYaxis()->SetRangeUser(0.f, 1.f);
    // eff_vs_pur->GetZaxis()->SetRangeUser(THRES_FROM, THRES_TO);
    eff_vs_pur->SetMarkerSize(0.4);
    eff_vs_pur->SetMarkerColor(kRed);
    eff_vs_pur->SetMarkerStyle(kFullSquare);


    // Init SV graphs
    TGraph* sv_roc_curve = new TGraph(SV_NTHRES, sv_fpr, sv_eff);
    sv_roc_curve->SetName("sv_roc_curve");
    sv_roc_curve->SetTitle("ROC curve");
    sv_roc_curve->GetXaxis()->SetTitle("FP Rate");
    sv_roc_curve->GetYaxis()->SetTitle("TP Rate");
    sv_roc_curve->GetXaxis()->SetRangeUser(0.f, 1.f);
    sv_roc_curve->GetYaxis()->SetRangeUser(0.f, 1.f);
    sv_roc_curve->SetMarkerColor(kBlack);
    sv_roc_curve->SetMarkerSize(1.4);
    sv_roc_curve->SetMarkerStyle(kFullStar);
    TGraph* sv_eff_vs_pur = new TGraph(SV_NTHRES, sv_eff, sv_pur);
    sv_eff_vs_pur->SetName("sv_eff_vs_pur");
    sv_eff_vs_pur->SetTitle("");
    sv_eff_vs_pur->GetXaxis()->SetTitle("Efficiency");
    sv_eff_vs_pur->GetYaxis()->SetTitle("Purity");
    sv_eff_vs_pur->GetXaxis()->SetRangeUser(0.f, 1.f);
    sv_eff_vs_pur->GetYaxis()->SetRangeUser(0.f, 1.f);
    sv_eff_vs_pur->SetMarkerSize(1.4);
    sv_eff_vs_pur->SetMarkerStyle(kFullStar);
    sv_eff_vs_pur->SetMarkerColor(kBlack);


    // Draw hist Db
    TCanvas* canvas_db_distr = new TCanvas();
    gPad->SetLogy();
    for (int f=b; f<=l; f++)
        hist_db[f]->Draw("HIST;SAME");
    
    TLegend* leg_db_distr = new TLegend(.7, .8, .8, .7);
    leg_db_distr->SetFillColorAlpha(kWhite, 0.f);
    leg_db_distr->SetBorderSize(0);
    for (int f=b; f<=l; f++)
        leg_db_distr->AddEntry(hist_db[f], TString::Format("%c-jet", flav_c[f]), "L");
    leg_db_distr->Draw();


    // Draw ROC curve
    TCanvas* canvas_roc = new TCanvas();
    TLine* line = new TLine(0., 0., 1., 1.);
    line->SetLineStyle(3);
    line->Draw();
    roc_curve->Draw("AL;SAME");
    sv_roc_curve->Draw("P;SAME");
    line->Draw();

    TLegend* leg_roc = new TLegend(.6, .25, .95, .17);
    leg_roc->SetFillColorAlpha(kWhite, 0.f);
    leg_roc->SetBorderSize(0);
    leg_roc->AddEntry(roc_curve, "ROC curve (b-jet)", "L");
    leg_roc->AddEntry(sv_roc_curve, "SV method", "P");
    leg_roc->Draw();

    // Calc AUC
    TGraph roc_auc = *roc_curve;
    roc_auc.SetName("roc_auc");
    roc_auc.AddPoint(1.f, 0.f);
    std::cout << TString::Format("AUC: %.5f", roc_auc.Integral()) << std::endl;


    // Draw Eff vs. Pur
    TCanvas* canvas_eff_vs_pur = new TCanvas();
    // eff_vs_pur->Draw("PCOL");
    eff_vs_pur->Draw("AP");
    sv_eff_vs_pur->Draw("P;SAME");

    TLegend* leg_eff_vs_pur = new TLegend(.6, .85, .95, .77);
    leg_eff_vs_pur->SetFillColorAlpha(kWhite, 0.f);
    leg_eff_vs_pur->SetBorderSize(0);
    leg_eff_vs_pur->AddEntry(eff_vs_pur, "B-jet tagging", "P");
    leg_eff_vs_pur->AddEntry(sv_eff_vs_pur, "SV method", "P");
    leg_eff_vs_pur->Draw();


    // // Export figures
    // canvas_db_distr->Print("images/db_distr.pdf");
    // canvas_roc->Print("images/roc_curve_db.pdf");
    // canvas_eff_vs_pur->Print("images/eff_vs_pur_db.pdf");


    // // Export ROC curve as .root file
    // TFile* tf_graph = new TFile(TString::Format("S2GClassifyJets/db_fc%03d.root", int(PARAM_F*1.e3f)), "RECREATE");
    // tf_graph->Add(roc_curve);
    // tf_graph->Add(eff_vs_pur);
    // tf_graph->Write();
}