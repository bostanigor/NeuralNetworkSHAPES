using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Windows.Forms;
using System.Collections;

namespace NeuralNetwork1
{
    /// <summary>
    /// Класс для хранения образа – входной массив сигналов на сенсорах, выходные сигналы сети, и прочее
    /// </summary>
    public class Sample
    {
        /// <summary>
        /// Входной вектор
        /// </summary>
        public double[] input = null;

        /// <summary>
        /// Выходной вектор, задаётся извне как результат распознавания
        /// </summary>
        public double[] output = null;

        /// <summary>
        /// Вектор ошибки, вычисляется по какой-нибудь хитрой формуле
        /// </summary>
        public double[] error = null;

        /// <summary>
        /// Действительный класс образа. Указывается учителем
        /// </summary>
        public FigureType actualClass;

        /// <summary>
        /// Распознанный класс - определяется после обработки
        /// </summary>
        public FigureType recognizedClass;

        /// <summary>
        /// Конструктор образа - на основе входных данных для сенсоров, при этом можно указать класс образа, или не указывать
        /// </summary>
        /// <param name="inputValues"></param>
        /// <param name="sampleClass"></param>
        public Sample(double[] inputValues, int classesCount, FigureType sampleClass = FigureType.Undef)
        {
            //  Клонируем массивчик
            input = (double[])inputValues.Clone();
            output = new double[classesCount];
            if (sampleClass != FigureType.Undef) output[(int)sampleClass] = 1;

            recognizedClass = FigureType.Undef;
            actualClass = sampleClass;
        }

        /// <summary>
        /// Обработка реакции сети на данный образ на основе вектора выходов сети
        /// </summary>
        public void processOutput()
        {
            if (error == null)
                error = new double[output.Length];

            //  Нам так-то выход не нужен, нужна ошибка и определённый класс
            recognizedClass = 0;
            for (int i = 0; i < output.Length; ++i)
            {
                error[i] = ((i == (int)actualClass ? 1 : 0) - output[i]);
                if (output[i] > output[(int)recognizedClass]) recognizedClass = (FigureType)i;
            }
        }

        /// <summary>
        /// Вычисленная суммарная квадратичная ошибка сети. Предполагается, что целевые выходы - 1 для верного, и 0 для остальных
        /// </summary>
        /// <returns></returns>
        public double EstimatedError()
        {
            double Result = 0;
            for (int i = 0; i < output.Length; ++i)
                Result += Math.Pow(error[i], 2);
            return Result;
        }

        /// <summary>
        /// Добавляет к аргументу ошибку, соответствующую данному образу (не квадратичную!!!)
        /// </summary>
        /// <param name="errorVector"></param>
        /// <returns></returns>
        public void updateErrorVector(double[] errorVector)
        {
            for (int i = 0; i < errorVector.Length; ++i)
                errorVector[i] += error[i];
        }

        /// <summary>
        /// Представление в виде строки
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            string result = "Sample decoding : " + actualClass.ToString() + "(" + ((int)actualClass).ToString() +
                            "); " + Environment.NewLine + "Input : ";
            for (int i = 0; i < input.Length; ++i) result += input[i].ToString() + "; ";
            result += Environment.NewLine + "Output : ";
            if (output == null) result += "null;";
            else
                for (int i = 0; i < output.Length; ++i)
                    result += output[i].ToString() + "; ";
            result += Environment.NewLine + "Error : ";

            if (error == null) result += "null;";
            else
                for (int i = 0; i < error.Length; ++i)
                    result += error[i].ToString() + "; ";
            result += Environment.NewLine + "Recognized : " + recognizedClass.ToString() + "(" +
                      ((int)recognizedClass).ToString() + "); " + Environment.NewLine;

            return result;
        }

        /// <summary>
        /// Правильно ли распознан образ
        /// </summary>
        /// <returns></returns>
        public bool Correct() { return actualClass == recognizedClass; }
    }

    /// <summary>
    /// Выборка образов. Могут быть как классифицированные (обучающая, тестовая выборки), так и не классифицированные (обработка)
    /// </summary>
    public class SamplesSet : IEnumerable
    {
        /// <summary>
        /// Накопленные обучающие образы
        /// </summary>
        public List<Sample> samples = new List<Sample>();

        /// <summary>
        /// Добавление образа к коллекции
        /// </summary>
        /// <param name="image"></param>
        public void AddSample(Sample image)
        {
            samples.Add(image);
        }

        public int Count { get { return samples.Count; } }

        public IEnumerator GetEnumerator()
        {
            return samples.GetEnumerator();
        }

        /// <summary>
        /// Реализация доступа по индексу
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        public Sample this[int i]
        {
            get { return samples[i]; }
            set { samples[i] = value; }
        }

        public double ErrorsCount()
        {
            double correct = 0;
            double wrong = 0;
            foreach (var sample in samples)
                if (sample.Correct()) ++correct; else ++wrong;
            return correct / (correct + wrong);
        }

        // Тут бы ещё сохранение в файл и чтение сделать, вообще классно было бы
    }

    public class NeuralNetwork : BaseNetwork
    {
        static Random rand = new Random();
        private class NeuralNode
        {
            public double Value { get; set; }
            public double Error { get; set; }
            private List<NeuralLink> prevLayer;
            private List<NeuralLink> nextLayer;

            Func<double, double> activationFunction;

            public NeuralNode(Func<double, double> activationFunction)
            {
                this.activationFunction = activationFunction;
                prevLayer = new List<NeuralLink>();
                nextLayer = new List<NeuralLink>();
            }

            public void LinkNextNode(NeuralNode next)
            {
                var link = new NeuralLink(this, next, 1.0);
                this.nextLayer.Add(link);
                next.prevLayer.Add(link);
            }
            public double Eval()
            {
                double sum = EvalInputSum();
                Value = activationFunction.Invoke(sum);
                return Value;
            }

            public void ReevalLinks(double learningRate)
            {
                Error = Error * Value * (1 - Value);
                foreach (var link in prevLayer)
                {
                    link.prevNode.Error += Error * link.Weight;
                    link.Weight += learningRate * Error * link.InputValue;
                }
            }

            private double EvalInputSum() =>
                prevLayer.Sum(link => link.OutputValue);
        }

        private class NeuralLink
        {
            public NeuralNode prevNode;
            public NeuralNode nextNode;
            public double Weight { get; set; }

            public double InputValue
            {
                get { return prevNode.Value; }
            }

            public double OutputValue
            {
                get { return prevNode.Value * Weight; }
            }

            public NeuralLink(NeuralNode prevNode, NeuralNode nextNode, double weight = 1)
            {                
                this.prevNode = prevNode;
                this.nextNode = nextNode;
                this.Weight = rand.NextDouble() - 1.0;
            }
        }

        private NeuralNode[][] nodeLayers;
        private double learningRate;

        public NeuralNetwork(int[] structure, double learningRate = 0.25)
        {
            this.learningRate = learningRate;
            Init(structure);
        }

        public void Init(int[] structure)
        {
            Func<double, double> func = (double x) => {
                return 1 / (1 + Math.Exp(-x));
            };


            //  Creating layer structure
            nodeLayers = new NeuralNode[structure.Length][];
            for (var i = 0; i < structure.Length; i++)
            {
                var layerCount = structure[i];
                nodeLayers[i] = new NeuralNode[layerCount];
                for (var j = 0; j < layerCount; j++)
                    nodeLayers[i][j] = new NeuralNode(func);
            }

            // Linking nodes in adjacent layers
            for (var i = 0; i < nodeLayers.Length - 1; i++)
            {
                var currentLayer = nodeLayers[i];
                var nextLayer = nodeLayers[i + 1];

                foreach (var node in currentLayer)
                    foreach (var nextNode in nextLayer)
                        node.LinkNextNode(nextNode);
            }
        }

        public override void ReInit(int[] structure, double initialLearningRate = 0.25)
        {
            learningRate = initialLearningRate;
            Init(structure);
        }
                
        public override int Train(Sample sample, bool parallel = true)
        {
            var errorThreshold = 0.2;
            for (int i = 0; i < 1000; i++)
            {
                Run(sample);
                if (sample.Correct() && sample.EstimatedError() < errorThreshold)
                    return i;

                ReevalWeights(sample);
            }
            throw new Exception("Too hard to train");
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochs_count, double acceptable_erorr, bool parallel = true)
        {
            var guessLevel = 0.0;
            while (epochs_count > 0)
            {
                foreach (var sample in samplesSet)
                    if (Train((Sample)sample) == 0)
                        guessLevel += 1;
                guessLevel /= samplesSet.samples.Count;
                if (guessLevel > acceptable_erorr) return guessLevel;
                epochs_count--;
            }
            
            return guessLevel;
        }                        

        public override FigureType Predict(Sample sample)
        {
            Run(sample);
            return sample.recognizedClass;
        }

        public override double TestOnDataSet(SamplesSet testSet)
        {
            throw new NotImplementedException();
        }

        public override double[] getOutput() =>        
            nodeLayers.Last().Select(node => node.Value).ToArray();        

        private void Run(Sample sample)
        {
            // Init first layer by sensors' values
            for (var i = 0; i < sample.input.Length; i++)                
                nodeLayers[0][i].Value = sample.input[i];

            for (var i = 1; i < nodeLayers.Length; i++)
                foreach (var node in nodeLayers[i])
                    node.Eval();

            // Copy output to sample
            var lastLayer = nodeLayers.Last();
            sample.output = new double[lastLayer.Length];
            for (var i = 0; i < lastLayer.Length; i++)
                sample.output[i] = lastLayer[i].Value;

            sample.processOutput();
        }

        private void ReevalWeights(Sample sample)
        {
            //  Reset Errors
            foreach (var layer in nodeLayers)
                foreach (var node in layer)
                    node.Error = 0.0;

            //  Init errors on last layer
            var lastLayer = nodeLayers.Last();
            for (int i = 0; i < lastLayer.Length; i++)
                lastLayer[i].Error = sample.error[i];

            foreach (var layer in nodeLayers.Reverse())
                foreach (var node in layer)
                    node.ReevalLinks(learningRate);            
        }
    }    
}