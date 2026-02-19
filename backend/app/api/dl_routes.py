"""
OptionQuant — Deep Learning API Routes (Enterprise)
════════════════════════════════════════════════════
Endpoints:
  POST /forecast          — Hybrid LSTM+Transformer pricing forecast
  POST /train             — Train LSTM on synthetic data
  POST /predict-volatility — DL-based volatility prediction
  POST /market-sentiment  — Transformer sentiment analysis
  GET  /status            — DL model status
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from .. import dl, pricing
from ..auth import UserRecord, get_current_user
from ..schemas import (
    DLForecastRequest,
    DLForecastResponse,
    DLTrainRequest,
    DLTrainResponse,
    PricingRequest,
    SentimentRequest,
    SentimentResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/dl", tags=["dl"])


@router.post("/forecast", response_model=DLForecastResponse)
async def dl_forecast(
    request: DLForecastRequest,
    _user: UserRecord = Depends(get_current_user),
) -> DLForecastResponse:
    """
    Hybrid deep learning forecast combining:
    - LSTM price prediction
    - Transformer sentiment analysis
    - Black-Scholes & Monte Carlo benchmarks
    - Residual learning for error correction
    """
    try:
        predictor = dl.get_predictor()
        forecast = predictor.predict(
            spot=request.spot,
            strike=request.strike,
            maturity=request.maturity,
            rate=request.rate,
            volatility=request.volatility,
            option_type=request.option_type,
            news_text=request.news_text,
        )
        return DLForecastResponse(
            model=forecast.model,
            forecast_price=forecast.forecast_price,
            forecast_vol=forecast.forecast_vol,
            residual=forecast.residual,
            confidence=forecast.confidence,
            lstm_prediction=forecast.lstm_prediction,
            transformer_sentiment=forecast.transformer_sentiment,
            benchmarks=forecast.details,
            details=forecast.details,
        )
    except Exception as e:
        logger.error("DL forecast error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"DL forecast error: {e}")


@router.post("/forecast-legacy")
async def dl_forecast_legacy(
    request: PricingRequest,
    _user: UserRecord = Depends(get_current_user),
) -> dict:
    """Legacy forecast endpoint (backward compatibility)."""
    try:
        inputs = pricing.PricingInputs(**request.model_dump())
        mc_price = pricing.monte_carlo_gbm(inputs, seed=42)
        bs_price = pricing.black_scholes(inputs)
        hybrid = dl.residual_learning(bs_price, mc_price)
        return {
            "model": hybrid.model,
            "forecast_price": hybrid.forecast_price,
            "forecast_vol": hybrid.forecast_vol,
            "residual": hybrid.residual,
            "benchmarks": {"mc": mc_price, "bs": bs_price},
        }
    except Exception as e:
        logger.error("Legacy forecast error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Forecast error: {e}")


@router.post("/train", response_model=DLTrainResponse)
async def dl_train(
    request: DLTrainRequest,
    _user: UserRecord = Depends(get_current_user),
) -> DLTrainResponse:
    """
    Train LSTM on synthetic market data.
    Also validates Transformer sentiment accuracy.
    """
    try:
        predictor = dl.get_predictor()

        # Run training in thread pool to avoid blocking the event loop
        result = await asyncio.to_thread(
            predictor.train_on_synthetic,
            spot=request.spot,
            volatility=request.volatility,
            rate=request.rate,
            n_days=request.n_days,
            seed=request.seed,
        )

        # Validate transformer
        transformer = predictor.transformer
        test_texts = [
            ("Markets rally on strong earnings", "bullish"),
            ("Recession fears grow sharply", "bearish"),
            ("Fed holds rates steady", "neutral"),
        ]
        correct = sum(
            1 for text, exp in test_texts
            if transformer.analyze(text).sentiment == exp
        )
        transformer_acc = correct / len(test_texts)

        return DLTrainResponse(
            lstm_epochs=result.epochs_trained,
            lstm_rmse=round(result.final_rmse, 6),
            lstm_r_squared=round(result.r_squared, 4),
            lstm_elapsed_ms=round(result.elapsed_ms, 2),
            transformer_accuracy=round(transformer_acc, 4),
            total_time_ms=round(result.elapsed_ms, 2),
            train_loss=result.train_loss[-20:],
            val_loss=result.val_loss[-20:],
            details={
                "final_mae": round(result.final_mae, 6),
                "n_predictions": len(result.predictions),
            },
        )
    except Exception as e:
        logger.error("DL training error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training error: {e}")


@router.post("/predict-volatility")
async def dl_predict_volatility(
    request: PricingRequest,
    _user: UserRecord = Depends(get_current_user),
) -> dict:
    """
    Deep learning volatility prediction using LSTM.
    Predicts future realized volatility from price dynamics.
    """
    try:
        predictor = dl.get_predictor()
        if not predictor.is_trained:
            # Auto-train with defaults (non-blocking)
            await asyncio.to_thread(
                predictor.train_on_synthetic,
                spot=request.spot,
                volatility=request.volatility,
                n_days=300,
            )

        # Generate price history and predict next value
        rng = np.random.default_rng(42)
        dt = 1.0 / 252
        prices = np.zeros(60)
        prices[0] = request.spot
        for i in range(1, 60):
            z = rng.standard_normal()
            prices[i] = prices[i-1] * np.exp(
                (request.rate - 0.5 * request.volatility**2) * dt
                + request.volatility * np.sqrt(dt) * z
            )

        predicted_price = predictor.lstm.predict(prices, lookback=30)

        # Implied volatility from price movement
        returns = np.diff(np.log(prices[-20:]))
        realized_vol = float(np.std(returns) * np.sqrt(252))
        predicted_vol = max(0.01, realized_vol * (1 + 0.1 * (predicted_price / prices[-1] - 1)))

        return {
            "predicted_volatility": round(predicted_vol, 6),
            "realized_volatility": round(realized_vol, 6),
            "predicted_price": round(predicted_price, 4),
            "current_price": round(float(prices[-1]), 4),
            "model": "lstm-volatility",
            "confidence": round(min(0.9, max(0.1, 1 - abs(predicted_price / prices[-1] - 1))), 4),
        }
    except Exception as e:
        logger.error("Vol prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Volatility prediction error: {e}")


@router.post("/market-sentiment", response_model=SentimentResponse)
async def market_sentiment(
    request: SentimentRequest,
    _user: UserRecord = Depends(get_current_user),
) -> SentimentResponse:
    """
    Transformer-based market sentiment analysis.
    Analyzes financial text and returns sentiment, confidence,
    attention weights, risk level, and estimated price impact.
    """
    try:
        predictor = dl.get_predictor()
        result = predictor.transformer.analyze(request.text)
        return SentimentResponse(
            sentiment=result.sentiment,
            score=result.score,
            confidence=result.confidence,
            attention_weights=[
                {"word": w, "weight": round(wt, 4)}
                for w, wt in result.attention_weights
            ],
            risk_level=result.risk_level,
            price_impact=result.price_impact,
            details=result.details,
        )
    except Exception as e:
        logger.error("Sentiment error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sentiment analysis error: {e}")


@router.get("/status")
async def dl_status(
    _user: UserRecord = Depends(get_current_user),
) -> dict:
    """Get DL model status."""
    predictor = dl.get_predictor()
    return {
        "lstm_trained": predictor.is_trained,
        "lstm_hidden_dim": predictor.lstm.hidden_dim,
        "lstm_layers": predictor.lstm.n_layers,
        "transformer_embed_dim": predictor.transformer.embed_dim,
        "transformer_heads": predictor.transformer.n_heads,
        "last_training": asdict(predictor.last_result) if predictor.last_result else None,
    }
