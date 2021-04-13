package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
)

func Set(ni, nh, no, lh int) *Mlp {
	rand.Seed(0)

	mlp := &Mlp{
		nodeInput:  ni + 1, //add bais node
		nodeHidden: nh + 1, //add bais node
		nodeOutput: no,
		layHidden:  lh,
	}

	//重みの初期化
	mlp.weiInput = MatrixInit(mlp.nodeInput, mlp.nodeHidden, 0)
	for n := 0; n < mlp.nodeInput; n++ {
		for w := 0; w < mlp.nodeHidden; w++ {
			mlp.weiInput[n][w] = Random(-1, 1)
		}
	}
	if mlp.layHidden > 1 { //隠れ層が２層以上存在するなら作成。
		mlp.weiHidden = Tensor3Init(mlp.layHidden-1, mlp.nodeHidden, mlp.nodeHidden, 0)
		for i := 0; i < mlp.layHidden-1; i++ {
			for j := 0; j < mlp.nodeHidden; j++ {
				for k := 0; k < mlp.nodeHidden; k++ {
					mlp.weiHidden[i][j][k] = Random(-1, 1)
				}
			}
		}
	}
	mlp.weiOutput = MatrixInit(mlp.nodeHidden, mlp.nodeOutput, 0)
	for n := 0; n < mlp.nodeHidden; n++ {
		for w := 0; w < mlp.nodeOutput; w++ {
			mlp.weiOutput[n][w] = Random(-1, 1)
		}
	}

	//ノード自身の値
	mlp.valInput = VectorInit(mlp.nodeInput, 1)
	mlp.valHidden = MatrixInit(mlp.layHidden, mlp.nodeHidden, 1)
	mlp.valOutput = VectorInit(mlp.nodeOutput, 1)

	fmt.Println("[+]Init Ok")
	//fmt.Printf("%+v\n", mlp)
	return mlp
}

func (mlp *Mlp) Train() {
	for i := 0; i < iteration; i++ {
		var e float64
		e = 0
		for _, p := range patterns {
			//出力層と教師数が異なるならアウト！
			if len(mlp.valOutput) != len(p[1]) {
				log.Println("[-]The number of labels are invalid. Don't match the number of outputnodes")
				return
			}
			//compute forward
			mlp.Update(p[0])
			//compute backward
			e += mlp.Backpropagate(p[1])
		}
		gred[i] = e
	}
}

//compute forward
func (mlp *Mlp) Update(inputs []float64) {
	//入力層のセット
	for i, val := range inputs {
		mlp.valInput[i] = val
	}

	var sum float64
	for i := 0; i < mlp.nodeHidden-1; i++ { //最上位隠れ層のみ別で計算
		sum = 0
		for j := 0; j < mlp.nodeInput; j++ {
			sum += mlp.weiInput[j][i] * mlp.valInput[j]
		}
		mlp.valHidden[0][i] = Sigmoid(sum)
	}

	if mlp.layHidden > 1 { //隠れ層が2層以上なら計算
		for l := 1; l < mlp.layHidden; l++ { //l=0は上記(入力層)の計算で格納済み
			sum = 0
			for n := 0; n < mlp.nodeHidden; n++ {
				for w := 0; w < mlp.nodeHidden; w++ {
					sum += mlp.weiHidden[l-1][w][n] * mlp.valHidden[l-1][n] //weiHiddenは隠れ層が２層以上で有効。１層のみならそもそも存在しない。
				}
				mlp.valHidden[l][n] = Sigmoid(sum)
			}
		}
	}

	for i := 0; i < mlp.nodeOutput; i++ {
		sum = 0
		for j := 0; j < mlp.nodeHidden; j++ {
			sum += mlp.weiOutput[j][i] * mlp.valHidden[mlp.layHidden-1][j]
		}
		mlp.valOutput[i] = Sigmoid(sum)
	}
}

//compute backward
func (mlp *Mlp) Backpropagate(label []float64) float64 {
	mlp.deltaHidden = MatrixInit(mlp.layHidden, mlp.nodeHidden, 0)
	mlp.deltaInput = VectorInit(mlp.nodeInput, 0)
	mlp.deltaOutput = VectorInit(mlp.nodeOutput, 0)

	for i := 0; i < mlp.nodeOutput; i++ {
		mlp.deltaOutput[i] = Dsigmoid(mlp.valOutput[i]) * (mlp.valOutput[i] - label[i])
	}

	//入力側(l-1)を固定しながら、出力層側(l)を回していく
	for i := 0; i < mlp.nodeHidden; i++ {
		for j := 0; j < mlp.nodeOutput; j++ {
			//mlp.deltaHidden[mlp.layHidden-1][i] += mlp.deltaOutput[j] * mlp.weiOutput[i][j]
			mlp.deltaHidden[mlp.layHidden-1][i] += mlp.deltaOutput[j] * Dsigmoid(mlp.valHidden[mlp.layHidden-1][i]) * mlp.weiOutput[i][j]
			mlp.weiOutput[i][j] -= lRate * mlp.deltaOutput[j] * mlp.valHidden[mlp.layHidden-1][i]
		}
	}

	//隠れ層が2層以上なら計算
	for l := mlp.layHidden - 1; l > 0; l-- {
		for i := 0; i < mlp.nodeHidden; i++ {
			for j := 0; j < mlp.nodeHidden; j++ {
				mlp.deltaHidden[l-1][i] += mlp.deltaHidden[l][j] * Dsigmoid(mlp.valHidden[l-1][i]) * mlp.weiHidden[l-1][i][j]
				mlp.weiHidden[l-1][i][j] -= lRate * mlp.deltaHidden[l][j] * mlp.valHidden[l-1][i]
			}
		}
	}

	for i := 0; i < mlp.nodeInput; i++ {
		for j := 0; j < mlp.nodeHidden; j++ {
			//mlp.weiInput[i][j] -= lRate * mlp.deltaHidden[0][j] * Dsigmoid(mlp.valHidden[0][j]) * mlp.valInput[i]
			mlp.weiInput[i][j] -= lRate * mlp.deltaHidden[0][j] * mlp.valInput[i]
		}
	}

	var e float64
	e = 0
	for i := 0; i < mlp.nodeOutput; i++ {
		e += math.Pow((mlp.valOutput[i] - label[i]), 2)
	}

	return e
}

func (mlp *Mlp) Test(valInput [][][]float64) {
	for _, p := range valInput {
		mlp.Update(p[0])
		fmt.Println(p[0], "->", mlp.valOutput, " : ", p[1])
	}
}
