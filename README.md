# ProyectoDI

Proyecto DI es un programa realizado C# mediante el entorno de desarrollo de Visual Studio 2017. Mediante el sistema 
de bases de datos Microsoft SQL Sever, se interactua con una base de datos para tratar la información de un Iris
data set.

## Planteamiento de trabajo

El sistema de trabajo se basa en una solución común ***ProyectoDI***, la cual engloba varios proyectos que tienen
como trabajo el consultar a diferentes bases de datos un conjunto de datos del Iris.

El objetivo de estos diferentes proyectos es referenciarlos desde la solución común recuperando desde éste una 
lista de evidencias para poder utilizarlos en el aprendizaje automático y la minería de datos. 

Para esto, se debe trabajar con la funcionalidad de Weka, software que contiene una colección de herramientas de 
visualización y algoritmos para análisis de datos y modelado predictivo, unidos a una interfaz gráfica de usuario para 
acceder fácilmente a sus funcionalidades. 

## Funcionalidad de AccesoSqlServer

Concretamente en mi grupo, formado por Eric y Jose, se ha trabajado con SqlServer, se ha creado la referencia al acceso 
a una Lista de Evidencias obtenidas de una tabla del gestor de bases de datos SqlServer. 

Posteriormente para adaptar esos datos a la funcionalidad de Weka se ha tenido que trabajar con unas librerías especiales
para poder tratar el lenguaje de programación de Java en Visual Studio donde se trabaja en C#. Esto se ha logrado mediante
***`IKVM.NET`*** , que es una implementación de Java para Mono y Microsoft .NET Framework.

Y por último se ha trabajado en la elaboración de un modelo para simular la funcionalidad del software de Weka para 
analizar esos datos de tipo Evidencia y poder realizar un modelo predictivo.

``` 
namespace ModelosWeka
{
    class Program
    {
        static void Main(string[] args)
        }
            cvdTest();
        }

        public static void cvdTest()
        {

            //string ruta = Environment.CurrentDirectory;

            //string archivo = ruta + "/iris.arff";

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
```

## Documentación

[Visual Studio ](https://visualstudio.microsoft.com/es/)

[Sql Server ](https://www.microsoft.com/es-es/sql-server/sql-server-2017)

[Weka](https://es.wikipedia.org/wiki/Weka_(aprendizaje_autom%C3%A1tico)

[IKVM](https://sourceforge.net/p/ikvm/mailman/message/32808306/)

[SQL Server Management Studio (SSMS)](https://docs.microsoft.com/es-es/sql/ssms/sql-server-management-studio-ssms?view=sql-server-2017)

[Machine Learning, Iris Data Set](https://archive.ics.uci.edu/ml/datasets/Iris)

[IKVM.NET](https://www.ikvm.net/)


