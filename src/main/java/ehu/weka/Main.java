package ehu.weka;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.*;
import weka.core.converters.ConverterUtils;
import weka.core.neighboursearch.LinearNNSearch;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.util.Random;

public class Main {
    public static void main(String[] args) throws Exception {

        //datuak kargatu
        String path = "C:\\Users\\jongo\\OneDrive\\Escritorio\\3.Praktika\\balance-scale.arff";
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(path);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes()-1);

        //Cross Validation erabiliko ez balitz:
        /*
        //Randomize
        Randomize filter = new Randomize();
        filter.setInputFormat(data);
        Instances RandomData = Filter.useFilter(data,filter);
        System.out.println("Data instantzia kopurua: "+data.numInstances()+"\n");

        //Train
        RemovePercentage filterRemoveTrain = new RemovePercentage();
        filterRemoveTrain.setInputFormat(RandomData);
        filterRemoveTrain.setPercentage(30);
        Instances train = Filter.useFilter(RandomData,filterRemoveTrain);
        train.setClassIndex(train.numAttributes() - 1);
        System.out.println("Train instantzia kopurua: "+train.numInstances()+"\n");

        //Test
        RemovePercentage filterRemoveTest = new RemovePercentage();
        filterRemoveTest.setInputFormat(RandomData);
        filterRemoveTest.setPercentage(30);
        filterRemoveTest.setInvertSelection(true);
        Instances test = Filter.useFilter(RandomData,filterRemoveTest);
        test.setClassIndex(test.numAttributes() - 1);
        System.out.println("\n\nTest instantzia kopurua: "+test.numInstances());
         */

        //Bilaketak eta pisuak kargatu
        LinearNNSearch manDistance =new LinearNNSearch();
        manDistance.setDistanceFunction(new ManhattanDistance());
        LinearNNSearch euDistance =new LinearNNSearch();
        manDistance.setDistanceFunction(new EuclideanDistance());
        LinearNNSearch cheDistance =new LinearNNSearch();
        manDistance.setDistanceFunction(new ChebyshevDistance());
        LinearNNSearch filDistance =new LinearNNSearch();
        manDistance.setDistanceFunction(new FilteredDistance());
        LinearNNSearch minkDistance =new LinearNNSearch();
        manDistance.setDistanceFunction(new MinkowskiDistance());
        LinearNNSearch[] distances = new LinearNNSearch[] {manDistance,euDistance,cheDistance,filDistance,minkDistance};

        SelectedTag[] tags = new SelectedTag[]{new SelectedTag(IBk.WEIGHT_NONE, IBk.TAGS_WEIGHTING),new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING), new SelectedTag(IBk.WEIGHT_SIMILARITY, IBk.TAGS_WEIGHTING)};

        //Bestelako parametroak
        Evaluation eval = null;
        IBk ibk = new IBk();
        Double maxf = 0.0;
        int maxk=0;
        LinearNNSearch maxd=null;
        SelectedTag maxw=null;


        //Prozesaketa
        for (int k=1; k<=data.numInstances();k++){
            ibk.setKNN(k);
            for(LinearNNSearch d : distances){
                ibk.setNearestNeighbourSearchAlgorithm(d);
                for(SelectedTag w : tags){
                    eval=new Evaluation(data);
                    ibk.setDistanceWeighting(w);
                    eval.crossValidateModel(ibk,data,10,new Random(1));
                    Double fmeasure = eval.fMeasure(0);
                    System.out.println("k = " + k + " distantzia = " + d + " eta pisua = " + w );
                    if(maxf<fmeasure){
                        maxf = fmeasure;
                        maxk=k;
                        maxd=d;
                        maxw=w;
                    }
                }
            }
        }
        System.out.println("EMAITZAK: k-ren balio optimoa = "+maxk+" da, "+maxd+" distantziarekin eta "+maxw+" pisuarekin non fmeasure = "+maxf+" den.");
    }
}
