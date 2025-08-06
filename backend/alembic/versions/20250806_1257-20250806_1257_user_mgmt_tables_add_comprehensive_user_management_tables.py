"""Add comprehensive user management tables

Revision ID: 20250806_1257_user_mgmt_tables
Revises: 79889599fb59
Create Date: 2025-08-06 12:57:10.984372

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '20250806_1257_user_mgmt_tables'
down_revision = '79889599fb59'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create user_sessions table
    op.create_table('user_sessions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('session_token', sa.String(length=255), nullable=False),
        sa.Column('refresh_token', sa.String(length=255), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('device_type', sa.String(length=50), nullable=True),
        sa.Column('browser', sa.String(length=100), nullable=True),
        sa.Column('platform', sa.String(length=100), nullable=True),
        sa.Column('country', sa.String(length=100), nullable=True),
        sa.Column('city', sa.String(length=100), nullable=True),
        sa.Column('timezone', sa.String(length=50), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_activity', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('login_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('logout_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('remember_me', sa.Boolean(), nullable=False),
        sa.Column('is_suspicious', sa.Boolean(), nullable=False),
        sa.Column('login_method', sa.String(length=50), nullable=False),
        sa.Column('two_factor_verified', sa.Boolean(), nullable=False),
        sa.Column('session_data', sa.Text(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_sessions_session_token'), 'user_sessions', ['session_token'], unique=True)
    op.create_index(op.f('ix_user_sessions_refresh_token'), 'user_sessions', ['refresh_token'], unique=True)
    op.create_index(op.f('ix_user_sessions_status'), 'user_sessions', ['status'], unique=False)
    op.create_index(op.f('ix_user_sessions_user_id'), 'user_sessions', ['user_id'], unique=False)

    # Create audit_logs table
    op.create_table('audit_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('event_type', sa.String(length=50), nullable=False),
        sa.Column('severity', sa.String(length=20), nullable=False),
        sa.Column('category', sa.String(length=50), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('session_id', sa.String(length=255), nullable=True),
        sa.Column('request_id', sa.String(length=100), nullable=True),
        sa.Column('resource_type', sa.String(length=50), nullable=True),
        sa.Column('resource_id', sa.String(length=100), nullable=True),
        sa.Column('old_values', sa.JSON(), nullable=True),
        sa.Column('new_values', sa.JSON(), nullable=True),
        sa.Column('country', sa.String(length=100), nullable=True),
        sa.Column('city', sa.String(length=100), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('service_name', sa.String(length=100), nullable=True),
        sa.Column('service_version', sa.String(length=50), nullable=True),
        sa.Column('environment', sa.String(length=50), nullable=True),
        sa.Column('correlation_id', sa.String(length=100), nullable=True),
        sa.Column('parent_event_id', sa.Integer(), nullable=True),
        sa.Column('risk_score', sa.Integer(), nullable=False),
        sa.Column('is_automated', sa.String(length=10), nullable=False),
        sa.Column('requires_review', sa.String(length=10), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_audit_logs_event_type'), 'audit_logs', ['event_type'], unique=False)
    op.create_index(op.f('ix_audit_logs_severity'), 'audit_logs', ['severity'], unique=False)
    op.create_index(op.f('ix_audit_logs_category'), 'audit_logs', ['category'], unique=False)
    op.create_index(op.f('ix_audit_logs_session_id'), 'audit_logs', ['session_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_request_id'), 'audit_logs', ['request_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_resource_id'), 'audit_logs', ['resource_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_timestamp'), 'audit_logs', ['timestamp'], unique=False)
    op.create_index(op.f('ix_audit_logs_correlation_id'), 'audit_logs', ['correlation_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_user_id'), 'audit_logs', ['user_id'], unique=False)

    # Create api_keys table
    op.create_table('api_keys',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('key_id', sa.String(length=64), nullable=False),
        sa.Column('key_hash', sa.String(length=128), nullable=False),
        sa.Column('key_prefix', sa.String(length=16), nullable=False),
        sa.Column('scopes', sa.JSON(), nullable=False),
        sa.Column('rate_limit_per_hour', sa.Integer(), nullable=False),
        sa.Column('rate_limit_per_day', sa.Integer(), nullable=False),
        sa.Column('allowed_ips', sa.JSON(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('usage_count', sa.Integer(), nullable=False),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_used_ip', sa.String(length=45), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('auto_renew', sa.Boolean(), nullable=False),
        sa.Column('created_by_ip', sa.String(length=45), nullable=True),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('revoked_reason', sa.Text(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_api_keys_key_id'), 'api_keys', ['key_id'], unique=True)
    op.create_index(op.f('ix_api_keys_status'), 'api_keys', ['status'], unique=False)
    op.create_index(op.f('ix_api_keys_user_id'), 'api_keys', ['user_id'], unique=False)

    # Create user_settings table
    op.create_table('user_settings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('theme', sa.String(length=20), nullable=False),
        sa.Column('language', sa.String(length=10), nullable=False),
        sa.Column('timezone', sa.String(length=50), nullable=False),
        sa.Column('date_format', sa.String(length=20), nullable=False),
        sa.Column('time_format', sa.String(length=10), nullable=False),
        sa.Column('default_chart_timeframe', sa.String(length=10), nullable=False),
        sa.Column('chart_type', sa.String(length=20), nullable=False),
        sa.Column('show_volume', sa.Boolean(), nullable=False),
        sa.Column('show_indicators', sa.Boolean(), nullable=False),
        sa.Column('chart_theme', sa.String(length=20), nullable=False),
        sa.Column('rsi_period', sa.Integer(), nullable=False),
        sa.Column('macd_fast', sa.Integer(), nullable=False),
        sa.Column('macd_slow', sa.Integer(), nullable=False),
        sa.Column('macd_signal', sa.Integer(), nullable=False),
        sa.Column('bb_period', sa.Integer(), nullable=False),
        sa.Column('bb_std', sa.Float(), nullable=False),
        sa.Column('default_position_size', sa.Float(), nullable=False),
        sa.Column('risk_per_trade', sa.Float(), nullable=False),
        sa.Column('auto_execute_signals', sa.Boolean(), nullable=False),
        sa.Column('require_confirmation', sa.Boolean(), nullable=False),
        sa.Column('preferred_exchanges', sa.JSON(), nullable=True),
        sa.Column('email_notifications', sa.Boolean(), nullable=False),
        sa.Column('push_notifications', sa.Boolean(), nullable=False),
        sa.Column('sms_notifications', sa.Boolean(), nullable=False),
        sa.Column('notify_new_signals', sa.Boolean(), nullable=False),
        sa.Column('notify_signal_updates', sa.Boolean(), nullable=False),
        sa.Column('notify_trade_executions', sa.Boolean(), nullable=False),
        sa.Column('notify_portfolio_changes', sa.Boolean(), nullable=False),
        sa.Column('notify_price_alerts', sa.Boolean(), nullable=False),
        sa.Column('notify_system_updates', sa.Boolean(), nullable=False),
        sa.Column('price_alert_frequency', sa.String(length=20), nullable=False),
        sa.Column('price_alert_threshold', sa.Float(), nullable=False),
        sa.Column('dashboard_layout', sa.JSON(), nullable=True),
        sa.Column('favorite_symbols', sa.JSON(), nullable=True),
        sa.Column('watchlist_symbols', sa.JSON(), nullable=True),
        sa.Column('hidden_sections', sa.JSON(), nullable=True),
        sa.Column('preferred_agents', sa.JSON(), nullable=True),
        sa.Column('agent_weights', sa.JSON(), nullable=True),
        sa.Column('min_signal_confidence', sa.Float(), nullable=False),
        sa.Column('api_rate_limit', sa.Integer(), nullable=False),
        sa.Column('api_access_level', sa.String(length=20), nullable=False),
        sa.Column('profile_public', sa.Boolean(), nullable=False),
        sa.Column('share_portfolio_performance', sa.Boolean(), nullable=False),
        sa.Column('allow_data_collection', sa.Boolean(), nullable=False),
        sa.Column('marketing_emails', sa.Boolean(), nullable=False),
        sa.Column('custom_settings', sa.JSON(), nullable=True),
        sa.Column('feature_flags', sa.JSON(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_settings_user_id'), 'user_settings', ['user_id'], unique=True)

    # Update portfolios table to add user relationship
    op.add_column('portfolios', sa.Column('user_id', sa.Integer(), nullable=False))
    op.create_foreign_key('fk_portfolios_user_id', 'portfolios', 'users', ['user_id'], ['id'])
    op.create_index(op.f('ix_portfolios_user_id'), 'portfolios', ['user_id'], unique=False)


def downgrade() -> None:
    # Drop indexes and foreign keys
    op.drop_index(op.f('ix_portfolios_user_id'), table_name='portfolios')
    op.drop_constraint('fk_portfolios_user_id', 'portfolios', type_='foreignkey')
    op.drop_column('portfolios', 'user_id')

    # Drop user_settings table
    op.drop_index(op.f('ix_user_settings_user_id'), table_name='user_settings')
    op.drop_table('user_settings')

    # Drop api_keys table
    op.drop_index(op.f('ix_api_keys_user_id'), table_name='api_keys')
    op.drop_index(op.f('ix_api_keys_status'), table_name='api_keys')
    op.drop_index(op.f('ix_api_keys_key_id'), table_name='api_keys')
    op.drop_table('api_keys')

    # Drop audit_logs table
    op.drop_index(op.f('ix_audit_logs_user_id'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_correlation_id'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_timestamp'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_resource_id'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_request_id'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_session_id'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_category'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_severity'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_event_type'), table_name='audit_logs')
    op.drop_table('audit_logs')

    # Drop user_sessions table
    op.drop_index(op.f('ix_user_sessions_user_id'), table_name='user_sessions')
    op.drop_index(op.f('ix_user_sessions_status'), table_name='user_sessions')
    op.drop_index(op.f('ix_user_sessions_refresh_token'), table_name='user_sessions')
    op.drop_index(op.f('ix_user_sessions_session_token'), table_name='user_sessions')
    op.drop_table('user_sessions')