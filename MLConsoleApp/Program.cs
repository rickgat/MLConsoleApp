// See https://aka.ms/new-console-template for more information
using System;
using Microsoft.ML;
using Microsoft.ML.Data;

class Program
{
    public class HouseData
    {
        public float Size {get; set;}
        public float Price {get; set;}
    }

    public class Prediction
    {
        [ColumnName("Score")]
        public float Price {get; set;}
    }

    static void Main(string[] args)
    {
        MLContext mLContext = new MLContext();

        HouseData[] houseData = {
            new HouseData() {Size = 1.1F, Price = 1.2F},
            new HouseData() {Size = 1.9F, Price = 2.3F},
            new HouseData() {Size = 2.8F, Price = 3.0F},
            new HouseData() {Size = 3.4F, Price = 3.7F}};
            IDataView trainingData = mLContext.Data.LoadFromEnumerable(houseData);

            var pipeline = mLContext.Transforms.Concatenate("Features", new[] {"Size"})
            .Append(mLContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 100));

            var model = pipeline.Fit(trainingData);

            var size = new HouseData() {Size = 2.5F};
            var price = mLContext.Model.CreatePredictionEngine<HouseData, Prediction>(model).Predict(size);

            System.Console.WriteLine($"Predicted price for size: {size.Size*1000} sq ft= {price.Price*100:C}k");
    }
}

