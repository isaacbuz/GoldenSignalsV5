"""Add AI hedge fund fields to signals

Revision ID: f73c17986ab5
Revises: 
Create Date: 2025-08-05 19:43:15.842082

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f73c17986ab5'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add AI hedge fund specific fields to signals table
    op.add_column('signals', sa.Column('predicted_price', sa.Float(), nullable=True))
    op.add_column('signals', sa.Column('prediction_timeframe', sa.String(20), nullable=True))
    op.add_column('signals', sa.Column('model_version', sa.String(50), nullable=True))
    op.add_column('signals', sa.Column('model_accuracy', sa.Float(), nullable=True))
    op.add_column('signals', sa.Column('feature_importance', sa.JSON(), nullable=True))
    op.add_column('signals', sa.Column('alternative_data_signals', sa.JSON(), nullable=True))
    op.add_column('signals', sa.Column('hedge_fund_strategy', sa.String(50), nullable=True))
    op.add_column('signals', sa.Column('sharpe_ratio', sa.Float(), nullable=True))
    op.add_column('signals', sa.Column('sortino_ratio', sa.Float(), nullable=True))
    op.add_column('signals', sa.Column('calmar_ratio', sa.Float(), nullable=True))
    op.add_column('signals', sa.Column('social_sentiment_score', sa.Float(), nullable=True))
    op.add_column('signals', sa.Column('news_sentiment_score', sa.Float(), nullable=True))
    op.add_column('signals', sa.Column('options_flow_signal', sa.String(50), nullable=True))
    op.add_column('signals', sa.Column('institutional_activity', sa.JSON(), nullable=True))
    op.add_column('signals', sa.Column('sector', sa.String(50), nullable=True))
    op.add_column('signals', sa.Column('correlation_analysis', sa.JSON(), nullable=True))
    op.add_column('signals', sa.Column('ml_models_used', sa.JSON(), nullable=True))
    op.add_column('signals', sa.Column('backtested_performance', sa.JSON(), nullable=True))


def downgrade() -> None:
    # Remove AI hedge fund specific fields
    op.drop_column('signals', 'backtested_performance')
    op.drop_column('signals', 'ml_models_used')
    op.drop_column('signals', 'correlation_analysis')
    op.drop_column('signals', 'sector')
    op.drop_column('signals', 'institutional_activity')
    op.drop_column('signals', 'options_flow_signal')
    op.drop_column('signals', 'news_sentiment_score')
    op.drop_column('signals', 'social_sentiment_score')
    op.drop_column('signals', 'calmar_ratio')
    op.drop_column('signals', 'sortino_ratio')
    op.drop_column('signals', 'sharpe_ratio')
    op.drop_column('signals', 'hedge_fund_strategy')
    op.drop_column('signals', 'alternative_data_signals')
    op.drop_column('signals', 'feature_importance')
    op.drop_column('signals', 'model_accuracy')
    op.drop_column('signals', 'model_version')
    op.drop_column('signals', 'prediction_timeframe')
    op.drop_column('signals', 'predicted_price')