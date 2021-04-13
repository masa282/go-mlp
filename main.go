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
	iteration = 1000
	lRate     = 0.6
)

var patterns = [][][]float64{
	{{0, 0}, {0}},
	{{0, 1}, {1}},
	{{1, 0}, {1}},
	{{1, 1}, {0}},
}

var gred = make([]float64, iteration)

func main() {
	mlp := Set(2, 2, 1, 1) //入力層ノード、隠れ層ノード、出力層ノード、隠れ層数
	mlp.Train()
	mlp.Test(patterns)

	//fmt.Println(gred)
	outputIMG()
	//行列積を作成すること。関数化
}

/*func run() {
	Mlp, labels, err := loadModel()
	if err != nil {
		log.Fatal(err)
	}
}*/

func outputIMG() {
	p, err := plot.New()
	if err != nil {
		log.Panic(err)
	}
	p.Title.Text = "MLP_Error"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	//bs, err := plotter.NewBubbles(plotter.XYZs{{xp[0], xp[1], 1}}, vg.Points(2), vg.Points(2))
	//bs, err := plotter.NewLine(plo)
	for i, v := range gred {
		bs, err := plotter.NewScatter(plotter.XYZs{{float64(i), v, 1}}) //, vg.Points(2), vg.Points(2))
		if err != nil {
			panic(err)
		}
		bs.Color = color.RGBA{R: 0, G: 0, B: 255, A: 255}
		p.Add(bs)
	}

	//plotutil.AddLinePoints(p, "", plotter.XYs{{0, plane(w, 0)}, {4, plane(w, 4)}})
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "mlp.png"); err != nil {
		panic(err)
	}
}
