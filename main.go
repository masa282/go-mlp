package main

import (
	"image/color"
	"log"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

//Multi Layer Perceptron
type Mlp struct {
	layHidden                         int
	nodeInput, nodeHidden, nodeOutput int
	weiInput, weiOutput               [][]float64
	weiHidden                         [][][]float64
	valInput, valOutput               []float64
	valHidden                         [][]float64
	deltaInput, deltaOutput           []float64
	deltaHidden                       [][]float64
}

const (
	iteration = 5000
	lRate     = 0.6
)

var gred = make([]float64, iteration)

func main() {
	//learning data / label
	var patterns = [][][]float64{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	}

	mlp := Set(2, 2, 1, 1) //入力層ノード、隠れ層ノード、出力層ノード、隠れ層数
	mlp.Train(patterns)
	mlp.Test(patterns)

	//fmt.Println(gred)
	outputIMG()
}

func outputIMG() {
	p, err := plot.New()
	if err != nil {
		log.Panic(err)
	}
	p.Title.Text = "MLP_Error"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	for i, v := range gred {
		bs, err := plotter.NewScatter(plotter.XYZs{{float64(i), v, 1}})
		if err != nil {
			panic(err)
		}
		bs.Color = color.RGBA{R: 0, G: 0, B: 255, A: 255}
		p.Add(bs)
	}

	if err := p.Save(4*vg.Inch, 4*vg.Inch, "mlp.png"); err != nil {
		panic(err)
	}
}
