import { evaluate, mean, std, concat, ones } from 'mathjs';
import React, { useEffect, useState, useCallback } from 'react';

class MultiLinearRegression {
    constructor() {
        // 빠르기;
        this.ALPHA = 0.1;
        // gradient 를 얼마만큼 움직여서 확인할 것인가?
        this.ITERATIONS = 200;
        this.mu = null;
        this.sigma = null;
        this.theta = null;
        this.isTrained = false;
    }

    execute(inputVector) {
        const { mu, sigma, theta } = this;
        let vectorizedInput = [1];
        for (let i = 0; i < inputVector.length; i++) {
            vectorizedInput.push((inputVector[i] - mu[i]) / sigma[i]);
        }
        const result = evaluate('vectorizedInput * theta', {
            vectorizedInput,
            theta,
        });
        return result;
    }

    train(matrix) {
        // large X means matrix version of x
        let X = evaluate(`matrix[:, 1:${matrix[0].length - 1}]`, {
            matrix,
        });
        let y = evaluate(`matrix[:,${matrix[0].length}]`, {
            matrix,
        });

        let m = y.length;

        let { XNorm, mu, sigma } = this.featureNormalize(X);
        this.mu = mu;
        this.sigma = sigma;

        // Gradient Descent
        XNorm = concat(ones([m, 1]).valueOf(), XNorm);

        let theta = Array(matrix[0].length)
            .fill()
            .map(() => {
                return [0];
            });
        this.theta = this.gradientDescentMulti(XNorm, y, theta);
        this.isTrained = true;
    }

    get alpha() {
        return this.ALPHA;
    }

    set alpha(value) {
        this.ALPHA = Number(value);
    }

    get iteration() {
        return this.ITERATIONS;
    }

    set iteration(value) {
        this.ITERATIONS = value;
    }

    gradientDescentMulti(X, y, theta) {
        const m = y.length;

        const { ALPHA, ITERATIONS } = this;
        for (let i = 0; i < ITERATIONS; i++) {
            theta = evaluate(`theta - ALPHA / m * ((X * theta - y)' * X)'`, {
                theta,
                ALPHA,
                m,
                X,
                y,
            });
        }
        return theta;
    }

    featureNormalize = (X) => {
        const mu = this.getMeanAsRowVector(X);
        const sigma = this.getStdAsRowVector(X);
        const n = X[0].length;
        for (let i = 0; i < n; i++) {
            let featureVector = evaluate(`X[:,${i + 1}]`, { X });
            let featureMeanVector = evaluate('featureVector - mu', {
                featureVector,
                mu: mu[i],
            });
            let normalizedVector = evaluate('featureMeanVector / sigma', {
                featureMeanVector,
                sigma: sigma[i],
            });
            evaluate(`X[:, ${i + 1}] = normalizedVector`, {
                X,
                normalizedVector,
            });
        }

        return { XNorm: X, mu, sigma };
    };

    getMeanAsRowVector = (matrix) => {
        const n = matrix[0].length;

        const vectors = Array(n)
            .fill()
            .map((_, i) => evaluate(`matrix[:, ${i + 1}]`, { matrix }));

        return vectors.reduce((result, vector) => result.concat(mean(vector)), []);
    };

    getStdAsRowVector = (matrix) => {
        const n = matrix[0].length;

        const vectors = Array(n)
            .fill()
            .map((_, i) => evaluate(`matrix[:, ${i + 1}]`, { matrix }));

        return vectors.reduce((result, vector) => result.concat(std(vector)), []);
    };
}

const Sample = (props) => {
    const [regressionManager] = useState(new MultiLinearRegression());
    const [arrayInput, setArrayInput] = useState('');
    const [testInput, setTestInput] = useState('');
    const [loadedFlag, setLoadedFlag] = useState('');
    const trainModel = () => {
        try {
            let result = arrayInput
                ? JSON.parse(arrayInput.toString())
                : [
                      [1, 1],
                      [2, 2],
                      [3, 3],
                      [4, 4],
                      [5, 5],
                      [6, 6],
                      [7, 7],
                      [8, 8],
                      [9, 9],
                  ];

            regressionManager.train(result);
            setLoadedFlag(arrayInput || '데이터 샘플');
        } catch (err) {
            console.log(err);
        }
    };

    useEffect(() => {}, []);

    return (
        <div>
            학습됨?
            <span style={{ fontSize: 20 }}>&nbsp;&nbsp;&nbsp; {loadedFlag ? 'READY' : 'NOPE'}</span>
            <hr />
            <div>
                현재 데이터 : {loadedFlag.toString()}
                <br />
                데이터 샘플 : [[1, 1],[2, 2],[3, 3],[4, 4],[5, 5],[6, 6],[7, 7],[8, 8],[9, 9]]
                <textarea
                    style={{ width: '100%', height: 100 }}
                    onBlur={(e) => {
                        setArrayInput(e.target.value);
                    }}
                ></textarea>
                <div style={{ float: 'left' }}>
                    스킵 간격 : &nbsp;
                    <input
                        onChange={(e) => (regressionManager.alpha = e.target.value)}
                        defaultValue={regressionManager.alpha}
                    ></input>
                </div>
                <div style={{ float: 'left' }}>
                    학습 횟수 : &nbsp;
                    <input
                        onChange={(e) => (regressionManager.iteration = e.target.value)}
                        defaultValue={regressionManager.iteration}
                    ></input>
                </div>
                <div style={{ width: '100%', height: 50 }}>
                    <div
                        style={{
                            width: 100,
                            height: 50,
                            border: 'solid',
                            backgroundColor: 'magenta',
                            float: 'right',
                        }}
                        onClick={() => trainModel()}
                    >
                        학습 시작하기
                    </div>
                </div>
            </div>
            <div>
                <div>
                    아래 값으로 테스트
                    <textarea
                        style={{ width: '100%', height: 100 }}
                        onBlur={(e) => {
                            setTestInput(e.target.value);
                        }}
                    ></textarea>
                    <div style={{ width: '100%', height: 50 }}>
                        <div
                            style={{
                                width: 100,
                                height: 50,
                                border: 'solid',
                                backgroundColor: 'cyan',
                                float: 'right',
                            }}
                            onClick={() => {
                                alert(regressionManager.execute(JSON.parse(testInput)));
                            }}
                        >
                            결과 확인
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Sample;
