package ehu.weka;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.ChebyshevDistance;
import weka.core.EuclideanDistance;
import weka.core.FilteredDistance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.MinkowskiDistance;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils;
import weka.core.neighboursearch.LinearNNSearch;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.util.Random;

public class Main {
    public static void main(String[] args) throws Exception {

        if (args.length != 1) {
            System.out.println("\nUsage: kNN </path/data.arff>\n");
        } else {
            //datuak kargatu
            //String path = "C:\\Users\\jongo\\OneDrive\\Escritorio\\3.Praktika\\balance-scale.arff";
            String path = args[0];
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(path);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

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
            LinearNNSearch manDistance = new LinearNNSearch();
            manDistance.setDistanceFunction(new ManhattanDistance());
            LinearNNSearch euDistance = new LinearNNSearch();
            euDistance.setDistanceFunction(new EuclideanDistance());
            LinearNNSearch cheDistance = new LinearNNSearch();
            cheDistance.setDistanceFunction(new ChebyshevDistance());
            LinearNNSearch filDistance = new LinearNNSearch();
            filDistance.setDistanceFunction(new FilteredDistance());
            LinearNNSearch minkDistance = new LinearNNSearch();
            minkDistance.setDistanceFunction(new MinkowskiDistance());
            LinearNNSearch[] distances = new LinearNNSearch[]{manDistance, euDistance, cheDistance, filDistance, minkDistance};

            SelectedTag[] tags = new SelectedTag[]{new SelectedTag(IBk.WEIGHT_NONE, IBk.TAGS_WEIGHTING), new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING), new SelectedTag(IBk.WEIGHT_SIMILARITY, IBk.TAGS_WEIGHTING)};

            //Bestelako parametroak
            Evaluation eval = null;
            Evaluation maxeval = new Evaluation(data);
            IBk ibk = new IBk();
            Double maxf = 0.0;
            int maxk = 0;
            LinearNNSearch maxd = null;
            SelectedTag maxw = null;
            int index = 0;
            int maxindex = 0;

            //Prozesaketa
            for (int k = 1; k <= data.numInstances()*0.9; k++) {
                ibk.setKNN(k);
                index = 0;
                for (LinearNNSearch d : distances) {
                    ibk.setNearestNeighbourSearchAlgorithm(d);
                    int pisua = 0;
                    for (SelectedTag w : tags) {
                        try {
                            eval = new Evaluation(data);
                            ibk.setDistanceWeighting(w);
                            eval.crossValidateModel(ibk, data, 10, new Random(1));
                            Double fmeasure = eval.weightedFMeasure();
                            System.out.println("k = " + k + " distantzia = " + printeatudistantzia(index) + " eta pisua = " +  printeatupisua(pisua));
                            pisua++;
                            if (maxf < fmeasure) {
                                maxf = fmeasure;
                                maxk = k;
                                maxd = d;
                                maxw = w;
                                maxindex = index;
                                maxeval = eval;
                            }
                        }
                        catch (Exception e) {
                            System.out.println("k = " + k + " distantzia = " + printeatudistantzia(index) + " eta pisua = " + printeatupisua(pisua) + " ERROREA");
                        }
                    }
                    index ++;
                }
            }
            System.out.println("EMAITZAK: k-ren balio optimoa = " + maxk + " da, " + printeatudistantzia(maxindex)+ " distantziarekin eta " + maxw + " pisuarekin non fmeasure = " + maxf + " den.");
            System.out.println(maxeval.toMatrixString());
        }
    }

    private static String printeatupisua(int pisua) {
        switch(pisua) {
            case 0:
                return "None";
            case 1:
                return "Inverse";
            case 2:
                return "Similarity";
        }
        return null;
    }

    private static String printeatudistantzia(int i) {
        String emaitza ="";
        switch (i){
            case 0:
                emaitza = "Manhattan Distance";
                break;
            case 1:
                emaitza = "Euclidean Distance";
                break;
            case 2:
                emaitza = "Chevishev Distance";
                break;
            case 3:
                emaitza = "Filtered Distance";
                break;
            case 4:
                emaitza = "Minkowski Distance";
                break;
            default:
                emaitza = "Manhattan Distance";
                break;
        }
        return emaitza;
    }
}
