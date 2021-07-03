package main

import (
	"math"
	"math/rand"
)

type Obj struct {
	data []float64
}

/*func (o *Obj) Init(d []float64) {
	o.size = len(d)
	if d == nil {
		o.data = make([]float64, o.size)
	} else {
		o.data = d
	}
}*/

func (o *Obj) Vector() []float64 {
	return o.data
}

func VectorInit(size int, n float64) []float64 {
	v := make([]float64, size)
	for i := 0; i < size; i++ {
		v[i] = n
	}
	return v
}

func (o *Obj) Matrix(m, n int) [][]float64 {
	if len(o.data) != m*n {
		return nil //error
	}
	mat := make([][]float64, len(o.data))
	for i := 0; i < m; i++ {
		mat[i] = make([]float64, n)
		ind := m * i
		for j := 0; j < n; j++ {
			mat[i][j] = o.data[ind+j]
		}
	}
	return mat
}

func MatrixInit(m, n int, v float64) [][]float64 {
	mat := make([][]float64, m)
	for i := 0; i < m; i++ {
		mat[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			mat[i][j] = v
		}
	}
	return mat
}

func Tensor3Init(l, m, n int, v float64) [][][]float64 {
	ten := make([][][]float64, l)
	for i := 0; i < l; i++ {
		ten[i] = make([][]float64, m)
		for j := 0; j < m; j++ {
			ten[i][j] = make([]float64, n)
			for k := 0; k < n; k++ {
				ten[i][j][k] = v
			}
		}
	}
	return ten
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func Dsigmoid(y float64) float64 {
	return y * (1 - y)
}

func Random(a, b float64) float64 {
	return (b-a)*rand.Float64() + a
}
