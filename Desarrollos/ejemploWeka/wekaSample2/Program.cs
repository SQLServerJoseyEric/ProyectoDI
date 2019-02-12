using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using java;
using ikvm;
using IKVM;
using IKVM.NativeCode.java;
using IKVM.Attributes;


namespace wekaSample2
{
    class Program
    {
        static void Main(string[] args)
        {

            cvdTest();

        }

        public static void cvdTest()
        {

            weka.core.Instances data = new weka.core.Instances(new java.io.FileReader(@"C:\Users\Jose\Desktop\iris.arff"));
            data.setClassIndex(data.numAttributes() - 1);

            weka.classifiers.Classifier cls = new weka.classifiers.bayes.NaiveBayes();

            int runs = 1;
            int folds = 10;

            // perform cross-validation
            for (int i = 0; i < runs; i++)
            {
                // randomize data
                int seed = i + 1;
                java.util.Random rand = new java.util.Random(seed);
                weka.core.Instances randData = new weka.core.Instances(data);
                randData.randomize(rand);
                if (randData.classAttribute().isNominal())
                    randData.stratify(folds);
                
                weka.classifiers.Evaluation eval = new weka.classifiers.Evaluation(randData);
                for (int n = 0; n < folds; n++)
                {
                    weka.core.Instances train = randData.trainCV(folds, n);
                    weka.core.Instances test = randData.testCV(folds, n);
                    // build and evaluate classifier
                    weka.classifiers.Classifier clsCopy = weka.classifiers.Classifier.makeCopy(cls);
                    clsCopy.buildClassifier(train);
                    eval.evaluateModel(clsCopy, test);
                }

               


                

            }
        }
    }
}
        

