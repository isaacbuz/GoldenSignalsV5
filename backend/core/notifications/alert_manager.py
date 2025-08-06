"""
Alerting and Notification System
Manages signal alerts, notifications, and communication channels
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import uuid

from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import select, and_, or_, func
from jinja2 import Template

from database.models import Alert, User
from database.connection import get_db
from core.events.bus import event_bus, EventTypes
from core.logging import get_logger
from agents.base import Signal

logger = get_logger(__name__)


class AlertType(Enum):
    """Types of alerts"""
    SIGNAL = "signal"  # Trading signal generated
    RISK = "risk"  # Risk limit breach
    PERFORMANCE = "performance"  # Agent performance alert
    SYSTEM = "system"  # System status alert
    MARKET = "market"  # Market condition alert
    POSITION = "position"  # Position alert (stop loss, take profit)
    OPPORTUNITY = "opportunity"  # Trading opportunity detected
    NEWS = "news"  # Important news alert


class AlertPriority(Enum):
    """Alert priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    """Notification delivery channels"""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    WEBSOCKET = "websocket"
    TELEGRAM = "telegram"
    SLACK = "slack"
    DISCORD = "discord"
    PUSH = "push"


class AlertStatus(Enum):
    """Alert status"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    ACKNOWLEDGED = "acknowledged"
    EXPIRED = "expired"


@dataclass
class AlertConfig:
    """Alert configuration"""
    alert_type: AlertType
    enabled: bool = True
    channels: List[NotificationChannel] = field(default_factory=list)
    min_priority: AlertPriority = AlertPriority.MEDIUM
    cooldown_minutes: int = 5  # Minimum time between similar alerts
    aggregate: bool = False  # Aggregate similar alerts
    aggregate_window: int = 60  # Seconds to aggregate
    filters: Dict[str, Any] = field(default_factory=dict)
    
    # Signal-specific settings
    min_confidence: float = 0.7
    signal_types: List[str] = field(default_factory=lambda: ["buy", "sell"])
    
    # Risk-specific settings
    risk_levels: List[str] = field(default_factory=lambda: ["high", "critical"])
    
    # Time restrictions
    active_hours: Optional[Dict[str, str]] = None  # {"start": "09:00", "end": "17:00"}
    active_days: List[int] = field(default_factory=lambda: list(range(7)))  # 0=Monday


class AlertNotification(BaseModel):
    """Alert notification model"""
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    alert_type: AlertType
    priority: AlertPriority
    title: str
    message: str
    
    # Metadata
    symbol: Optional[str] = None
    signal: Optional[Signal] = None
    agent_id: Optional[str] = None
    portfolio_id: Optional[int] = None
    
    # Additional data
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    # Delivery
    channels: List[NotificationChannel] = Field(default_factory=list)
    recipients: List[str] = Field(default_factory=list)
    status: AlertStatus = AlertStatus.PENDING
    
    # Tracking
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    delivery_attempts: int = 0
    error_message: Optional[str] = None


class NotificationTemplate(BaseModel):
    """Notification template"""
    template_id: str
    name: str
    alert_type: AlertType
    channel: NotificationChannel
    subject_template: str
    body_template: str
    format: str = "text"  # text, html, markdown
    variables: List[str] = Field(default_factory=list)


class ChannelConfig(BaseModel):
    """Channel-specific configuration"""
    channel: NotificationChannel
    enabled: bool = True
    
    # Email config
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    from_email: Optional[EmailStr] = None
    
    # Webhook config
    webhook_url: Optional[str] = None
    webhook_headers: Dict[str, str] = Field(default_factory=dict)
    
    # Telegram config
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    
    # Slack config
    slack_webhook: Optional[str] = None
    slack_channel: Optional[str] = None
    
    # Discord config
    discord_webhook: Optional[str] = None


class AlertManager:
    """
    Centralized alert and notification management
    """
    
    def __init__(self):
        self.alert_configs: Dict[AlertType, AlertConfig] = {}
        self.channel_configs: Dict[NotificationChannel, ChannelConfig] = {}
        self.templates: Dict[str, NotificationTemplate] = {}
        self.alert_queue: asyncio.Queue = asyncio.Queue()
        self.alert_history: List[AlertNotification] = []
        self.last_alert_times: Dict[str, datetime] = {}
        self.aggregation_buffers: Dict[str, List[AlertNotification]] = {}
        
        self._processing_active = False
        self._processor_task: Optional[asyncio.Task] = None
        
        # WebSocket connections for real-time alerts
        self.websocket_connections: List[Any] = []
        
        # Initialize default configurations
        self._initialize_defaults()
        
        logger.info("Alert Manager initialized")
    
    def _initialize_defaults(self) -> None:
        """Initialize default alert configurations"""
        # Signal alerts
        self.alert_configs[AlertType.SIGNAL] = AlertConfig(
            alert_type=AlertType.SIGNAL,
            enabled=True,
            channels=[NotificationChannel.WEBSOCKET, NotificationChannel.EMAIL],
            min_priority=AlertPriority.MEDIUM,
            min_confidence=0.7,
            cooldown_minutes=1
        )
        
        # Risk alerts
        self.alert_configs[AlertType.RISK] = AlertConfig(
            alert_type=AlertType.RISK,
            enabled=True,
            channels=[NotificationChannel.WEBSOCKET, NotificationChannel.EMAIL],
            min_priority=AlertPriority.HIGH,
            cooldown_minutes=5
        )
        
        # Performance alerts
        self.alert_configs[AlertType.PERFORMANCE] = AlertConfig(
            alert_type=AlertType.PERFORMANCE,
            enabled=True,
            channels=[NotificationChannel.WEBSOCKET],
            min_priority=AlertPriority.MEDIUM,
            cooldown_minutes=60,
            aggregate=True,
            aggregate_window=300
        )
        
        # System alerts
        self.alert_configs[AlertType.SYSTEM] = AlertConfig(
            alert_type=AlertType.SYSTEM,
            enabled=True,
            channels=[NotificationChannel.WEBSOCKET, NotificationChannel.EMAIL],
            min_priority=AlertPriority.HIGH,
            cooldown_minutes=15
        )
        
        # Initialize default templates
        self._initialize_templates()
    
    def _initialize_templates(self) -> None:
        """Initialize notification templates"""
        # Signal alert template
        self.templates["signal_email"] = NotificationTemplate(
            template_id="signal_email",
            name="Signal Email Alert",
            alert_type=AlertType.SIGNAL,
            channel=NotificationChannel.EMAIL,
            subject_template="🚨 Trading Signal: {{ signal.action|upper }} {{ symbol }}",
            body_template="""
            Trading Signal Alert
            
            Symbol: {{ symbol }}
            Action: {{ signal.action|upper }}
            Confidence: {{ (signal.confidence * 100)|round(1) }}%
            {% if signal.entry_price %}Entry Price: ${{ signal.entry_price|round(2) }}{% endif %}
            {% if signal.stop_loss %}Stop Loss: ${{ signal.stop_loss|round(2) }}{% endif %}
            {% if signal.take_profit %}Take Profit: ${{ signal.take_profit|round(2) }}{% endif %}
            
            {% if signal.reasoning %}Reasoning: {{ signal.reasoning }}{% endif %}
            
            Generated by: {{ agent_id }}
            Time: {{ created_at }}
            
            ⚠️ This is not financial advice. Please do your own research.
            """,
            format="text",
            variables=["symbol", "signal", "agent_id", "created_at"]
        )
        
        # Risk alert template
        self.templates["risk_email"] = NotificationTemplate(
            template_id="risk_email",
            name="Risk Alert Email",
            alert_type=AlertType.RISK,
            channel=NotificationChannel.EMAIL,
            subject_template="⚠️ Risk Alert: {{ title }}",
            body_template="""
            Risk Management Alert
            
            {{ message }}
            
            Risk Level: {{ priority|upper }}
            {% if portfolio_id %}Portfolio: #{{ portfolio_id }}{% endif %}
            
            {% if data.violations %}
            Violations:
            {% for violation in data.violations %}
            - {{ violation }}
            {% endfor %}
            {% endif %}
            
            {% if data.recommended_action %}
            Recommended Action: {{ data.recommended_action }}
            {% endif %}
            
            Time: {{ created_at }}
            """,
            format="text",
            variables=["title", "message", "priority", "portfolio_id", "data", "created_at"]
        )
        
        # WebSocket template (JSON)
        self.templates["signal_websocket"] = NotificationTemplate(
            template_id="signal_websocket",
            name="Signal WebSocket",
            alert_type=AlertType.SIGNAL,
            channel=NotificationChannel.WEBSOCKET,
            subject_template="",
            body_template='{{data|tojson}}',
            format="json"
        )
    
    async def send_alert(
        self,
        alert_type: AlertType,
        priority: AlertPriority,
        title: str,
        message: str,
        **kwargs
    ) -> str:
        """
        Send an alert
        
        Args:
            alert_type: Type of alert
            priority: Alert priority
            title: Alert title
            message: Alert message
            **kwargs: Additional alert data
            
        Returns:
            Alert ID
        """
        try:
            # Check if alert type is enabled
            config = self.alert_configs.get(alert_type)
            if not config or not config.enabled:
                logger.debug(f"Alert type {alert_type.value} is disabled")
                return ""
            
            # Check priority threshold
            if priority.value < config.min_priority.value:
                logger.debug(f"Alert priority {priority.value} below threshold")
                return ""
            
            # Check cooldown
            if not self._check_cooldown(alert_type, kwargs.get("symbol"), config.cooldown_minutes):
                logger.debug(f"Alert on cooldown for {alert_type.value}")
                return ""
            
            # Check time restrictions
            if not self._check_time_restrictions(config):
                logger.debug(f"Alert outside active hours")
                return ""
            
            # Create alert notification
            alert = AlertNotification(
                alert_type=alert_type,
                priority=priority,
                title=title,
                message=message,
                symbol=kwargs.get("symbol"),
                signal=kwargs.get("signal"),
                agent_id=kwargs.get("agent_id"),
                portfolio_id=kwargs.get("portfolio_id"),
                data=kwargs.get("data", {}),
                metadata=kwargs.get("metadata", {}),
                channels=config.channels,
                recipients=kwargs.get("recipients", [])
            )
            
            # Set expiration
            if priority == AlertPriority.URGENT:
                alert.expires_at = datetime.now() + timedelta(minutes=30)
            elif priority == AlertPriority.CRITICAL:
                alert.expires_at = datetime.now() + timedelta(hours=1)
            
            # Check if should aggregate
            if config.aggregate:
                await self._add_to_aggregation_buffer(alert, config)
            else:
                # Add to queue for immediate processing
                await self.alert_queue.put(alert)
            
            # Store in history
            self.alert_history.append(alert)
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]
            
            # Store in database
            await self._store_alert(alert)
            
            # Update last alert time
            key = f"{alert_type.value}_{kwargs.get('symbol', 'general')}"
            self.last_alert_times[key] = datetime.now()
            
            logger.info(f"Alert created: {alert.alert_id} - {title}")
            return alert.alert_id
            
        except Exception as e:
            logger.error(f"Failed to send alert: {str(e)}")
            return ""
    
    async def send_signal_alert(self, signal: Signal, agent_id: str) -> str:
        """
        Send trading signal alert
        
        Args:
            signal: Trading signal
            agent_id: Agent that generated signal
            
        Returns:
            Alert ID
        """
        config = self.alert_configs.get(AlertType.SIGNAL)
        
        # Check signal confidence
        if config and signal.confidence < config.min_confidence:
            logger.debug(f"Signal confidence {signal.confidence} below threshold")
            return ""
        
        # Check signal type
        if config and signal.action not in config.signal_types:
            logger.debug(f"Signal type {signal.action} not in configured types")
            return ""
        
        # Determine priority based on confidence
        if signal.confidence >= 0.9:
            priority = AlertPriority.URGENT
        elif signal.confidence >= 0.8:
            priority = AlertPriority.HIGH
        elif signal.confidence >= 0.7:
            priority = AlertPriority.MEDIUM
        else:
            priority = AlertPriority.LOW
        
        title = f"Signal: {signal.action.upper()} {signal.symbol}"
        message = f"Trading signal with {signal.confidence*100:.1f}% confidence"
        
        return await self.send_alert(
            AlertType.SIGNAL,
            priority,
            title,
            message,
            symbol=signal.symbol,
            signal=signal,
            agent_id=agent_id,
            data={
                "action": signal.action,
                "confidence": signal.confidence,
                "entry_price": signal.metadata.get("entry_price"),
                "stop_loss": signal.metadata.get("stop_loss"),
                "take_profit": signal.metadata.get("take_profit")
            }
        )
    
    async def process_alerts(self) -> None:
        """
        Process alert queue
        """
        while self._processing_active:
            try:
                # Get alert from queue with timeout
                alert = await asyncio.wait_for(
                    self.alert_queue.get(),
                    timeout=1.0
                )
                
                # Check if expired
                if alert.expires_at and datetime.now() > alert.expires_at:
                    alert.status = AlertStatus.EXPIRED
                    logger.debug(f"Alert {alert.alert_id} expired")
                    continue
                
                # Process each channel
                for channel in alert.channels:
                    try:
                        await self._send_to_channel(alert, channel)
                    except Exception as e:
                        logger.error(f"Failed to send to {channel.value}: {str(e)}")
                        alert.error_message = str(e)
                
                # Update status
                if alert.delivery_attempts > 0:
                    alert.status = AlertStatus.SENT
                    alert.sent_at = datetime.now()
                
                # Update in database
                await self._update_alert_status(alert)
                
            except asyncio.TimeoutError:
                # Process aggregation buffers
                await self._process_aggregation_buffers()
                continue
            except Exception as e:
                logger.error(f"Alert processing error: {str(e)}")
    
    async def _send_to_channel(
        self,
        alert: AlertNotification,
        channel: NotificationChannel
    ) -> None:
        """
        Send alert to specific channel
        
        Args:
            alert: Alert to send
            channel: Notification channel
        """
        config = self.channel_configs.get(channel)
        if not config or not config.enabled:
            logger.debug(f"Channel {channel.value} not configured or disabled")
            return
        
        alert.delivery_attempts += 1
        
        try:
            if channel == NotificationChannel.EMAIL:
                await self._send_email(alert, config)
            elif channel == NotificationChannel.WEBSOCKET:
                await self._send_websocket(alert)
            elif channel == NotificationChannel.WEBHOOK:
                await self._send_webhook(alert, config)
            elif channel == NotificationChannel.TELEGRAM:
                await self._send_telegram(alert, config)
            elif channel == NotificationChannel.SLACK:
                await self._send_slack(alert, config)
            elif channel == NotificationChannel.DISCORD:
                await self._send_discord(alert, config)
            else:
                logger.warning(f"Channel {channel.value} not implemented")
                
        except Exception as e:
            logger.error(f"Failed to send via {channel.value}: {str(e)}")
            raise
    
    async def _send_email(self, alert: AlertNotification, config: ChannelConfig) -> None:
        """
        Send email notification
        """
        if not config.smtp_host or not config.from_email:
            logger.error("Email configuration incomplete")
            return
        
        # Get template
        template = self.templates.get(f"{alert.alert_type.value}_email")
        if not template:
            template = self.templates.get("signal_email")  # Default
        
        # Render template
        subject = self._render_template(template.subject_template, alert)
        body = self._render_template(template.body_template, alert)
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = config.from_email
        msg['To'] = ', '.join(alert.recipients) if alert.recipients else config.from_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        try:
            with smtplib.SMTP(config.smtp_host, config.smtp_port) as server:
                if config.smtp_username and config.smtp_password:
                    server.starttls()
                    server.login(config.smtp_username, config.smtp_password)
                server.send_message(msg)
                
            logger.info(f"Email sent for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Email send failed: {str(e)}")
            raise
    
    async def _send_websocket(self, alert: AlertNotification) -> None:
        """
        Send WebSocket notification
        """
        # Prepare WebSocket message
        ws_message = {
            "type": "alert",
            "alert_type": alert.alert_type.value,
            "priority": alert.priority.value,
            "title": alert.title,
            "message": alert.message,
            "symbol": alert.symbol,
            "data": alert.data,
            "timestamp": alert.created_at.isoformat()
        }
        
        # Send to all connected WebSocket clients
        disconnected = []
        for ws in self.websocket_connections:
            try:
                await ws.send_json(ws_message)
            except Exception:
                disconnected.append(ws)
        
        # Remove disconnected clients
        for ws in disconnected:
            self.websocket_connections.remove(ws)
        
        # Also publish to event bus for other listeners
        await event_bus.publish(
            "alert.sent",
            data=ws_message
        )
    
    async def _send_webhook(self, alert: AlertNotification, config: ChannelConfig) -> None:
        """
        Send webhook notification
        """
        if not config.webhook_url:
            logger.error("Webhook URL not configured")
            return
        
        payload = {
            "alert_id": alert.alert_id,
            "type": alert.alert_type.value,
            "priority": alert.priority.value,
            "title": alert.title,
            "message": alert.message,
            "symbol": alert.symbol,
            "data": alert.data,
            "timestamp": alert.created_at.isoformat()
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                config.webhook_url,
                json=payload,
                headers=config.webhook_headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"Webhook returned {response.status}")
    
    async def _send_telegram(self, alert: AlertNotification, config: ChannelConfig) -> None:
        """
        Send Telegram notification
        """
        if not config.telegram_token or not config.telegram_chat_id:
            logger.error("Telegram configuration incomplete")
            return
        
        # Format message for Telegram
        text = f"""
🚨 *{alert.title}*

{alert.message}

Priority: {alert.priority.value.upper()}
{f'Symbol: {alert.symbol}' if alert.symbol else ''}
Time: {alert.created_at.strftime('%H:%M:%S')}
        """.strip()
        
        url = f"https://api.telegram.org/bot{config.telegram_token}/sendMessage"
        payload = {
            "chat_id": config.telegram_chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Telegram API returned {response.status}")
    
    async def _send_slack(self, alert: AlertNotification, config: ChannelConfig) -> None:
        """
        Send Slack notification
        """
        if not config.slack_webhook:
            logger.error("Slack webhook not configured")
            return
        
        # Format for Slack
        payload = {
            "text": alert.title,
            "attachments": [{
                "color": self._get_color_for_priority(alert.priority),
                "fields": [
                    {"title": "Message", "value": alert.message, "short": False},
                    {"title": "Priority", "value": alert.priority.value, "short": True},
                    {"title": "Type", "value": alert.alert_type.value, "short": True}
                ],
                "footer": "GoldenSignals AI",
                "ts": int(alert.created_at.timestamp())
            }]
        }
        
        if alert.symbol:
            payload["attachments"][0]["fields"].append(
                {"title": "Symbol", "value": alert.symbol, "short": True}
            )
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config.slack_webhook, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Slack webhook returned {response.status}")
    
    async def _send_discord(self, alert: AlertNotification, config: ChannelConfig) -> None:
        """
        Send Discord notification
        """
        if not config.discord_webhook:
            logger.error("Discord webhook not configured")
            return
        
        # Format for Discord
        embed = {
            "title": alert.title,
            "description": alert.message,
            "color": self._get_color_code_for_priority(alert.priority),
            "fields": [
                {"name": "Priority", "value": alert.priority.value.upper(), "inline": True},
                {"name": "Type", "value": alert.alert_type.value, "inline": True}
            ],
            "timestamp": alert.created_at.isoformat(),
            "footer": {"text": "GoldenSignals AI"}
        }
        
        if alert.symbol:
            embed["fields"].append(
                {"name": "Symbol", "value": alert.symbol, "inline": True}
            )
        
        payload = {"embeds": [embed]}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config.discord_webhook, json=payload) as response:
                if response.status not in [200, 204]:
                    raise Exception(f"Discord webhook returned {response.status}")
    
    def _render_template(self, template_str: str, alert: AlertNotification) -> str:
        """
        Render template with alert data
        """
        from jinja2 import Template
        
        template = Template(template_str)
        
        context = {
            "alert_id": alert.alert_id,
            "alert_type": alert.alert_type.value,
            "priority": alert.priority.value,
            "title": alert.title,
            "message": alert.message,
            "symbol": alert.symbol,
            "signal": alert.signal,
            "agent_id": alert.agent_id,
            "portfolio_id": alert.portfolio_id,
            "data": alert.data,
            "metadata": alert.metadata,
            "created_at": alert.created_at
        }
        
        return template.render(**context)
    
    def _get_color_for_priority(self, priority: AlertPriority) -> str:
        """
        Get color for priority (Slack format)
        """
        colors = {
            AlertPriority.LOW: "#36a64f",
            AlertPriority.MEDIUM: "#3aa3e3",
            AlertPriority.HIGH: "#ff9900",
            AlertPriority.URGENT: "#ff6600",
            AlertPriority.CRITICAL: "#ff0000"
        }
        return colors.get(priority, "#808080")
    
    def _get_color_code_for_priority(self, priority: AlertPriority) -> int:
        """
        Get color code for priority (Discord format)
        """
        colors = {
            AlertPriority.LOW: 0x36a64f,
            AlertPriority.MEDIUM: 0x3aa3e3,
            AlertPriority.HIGH: 0xff9900,
            AlertPriority.URGENT: 0xff6600,
            AlertPriority.CRITICAL: 0xff0000
        }
        return colors.get(priority, 0x808080)
    
    def _check_cooldown(self, alert_type: AlertType, symbol: Optional[str], cooldown_minutes: int) -> bool:
        """
        Check if alert is on cooldown
        """
        key = f"{alert_type.value}_{symbol or 'general'}"
        last_time = self.last_alert_times.get(key)
        
        if not last_time:
            return True
        
        time_since = datetime.now() - last_time
        return time_since.total_seconds() >= cooldown_minutes * 60
    
    def _check_time_restrictions(self, config: AlertConfig) -> bool:
        """
        Check if current time is within active hours
        """
        if not config.active_hours:
            return True
        
        now = datetime.now()
        current_day = now.weekday()
        
        # Check day restriction
        if current_day not in config.active_days:
            return False
        
        # Check hour restriction
        current_time = now.strftime("%H:%M")
        start_time = config.active_hours.get("start", "00:00")
        end_time = config.active_hours.get("end", "23:59")
        
        return start_time <= current_time <= end_time
    
    async def _add_to_aggregation_buffer(
        self,
        alert: AlertNotification,
        config: AlertConfig
    ) -> None:
        """
        Add alert to aggregation buffer
        """
        key = f"{alert.alert_type.value}_{alert.priority.value}"
        
        if key not in self.aggregation_buffers:
            self.aggregation_buffers[key] = []
        
        self.aggregation_buffers[key].append(alert)
    
    async def _process_aggregation_buffers(self) -> None:
        """
        Process aggregation buffers and send aggregated alerts
        """
        for key, alerts in list(self.aggregation_buffers.items()):
            if not alerts:
                continue
            
            # Check if buffer is old enough
            oldest = min(alert.created_at for alert in alerts)
            if (datetime.now() - oldest).total_seconds() < 60:
                continue
            
            # Create aggregated alert
            alert_type = alerts[0].alert_type
            priority = alerts[0].priority
            
            title = f"Aggregated {alert_type.value} alerts ({len(alerts)} total)"
            message = f"Multiple {alert_type.value} alerts triggered"
            
            aggregated = AlertNotification(
                alert_type=alert_type,
                priority=priority,
                title=title,
                message=message,
                data={"aggregated_alerts": [a.alert_id for a in alerts]},
                channels=alerts[0].channels
            )
            
            # Send aggregated alert
            await self.alert_queue.put(aggregated)
            
            # Clear buffer
            self.aggregation_buffers[key] = []
    
    async def _store_alert(self, alert: AlertNotification) -> None:
        """
        Store alert in database
        """
        try:
            async with get_db() as session:
                db_alert = Alert(
                    alert_type=alert.alert_type.value,
                    severity=alert.priority.value,
                    source=alert.agent_id or "system",
                    title=alert.title,
                    message=alert.message,
                    metadata={
                        "alert_id": alert.alert_id,
                        "symbol": alert.symbol,
                        "data": alert.data,
                        "channels": [c.value for c in alert.channels]
                    }
                )
                session.add(db_alert)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to store alert in database: {str(e)}")
    
    async def _update_alert_status(self, alert: AlertNotification) -> None:
        """
        Update alert status in database
        """
        # Implementation would update the alert record
        pass
    
    async def start_processing(self) -> None:
        """
        Start alert processing
        """
        if self._processing_active:
            return
        
        self._processing_active = True
        self._processor_task = asyncio.create_task(self.process_alerts())
        
        # Subscribe to events
        await event_bus.subscribe(EventTypes.SIGNAL_GENERATED, self._on_signal_generated)
        await event_bus.subscribe("risk.alert", self._on_risk_alert)
        await event_bus.subscribe("performance.alert", self._on_performance_alert)
        
        logger.info("Alert processing started")
    
    async def stop_processing(self) -> None:
        """
        Stop alert processing
        """
        self._processing_active = False
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Alert processing stopped")
    
    def set_alert_config(self, alert_type: AlertType, config: AlertConfig) -> None:
        """
        Set alert configuration
        """
        self.alert_configs[alert_type] = config
        logger.info(f"Updated config for {alert_type.value}")
    
    def set_channel_config(self, channel: NotificationChannel, config: ChannelConfig) -> None:
        """
        Set channel configuration
        """
        self.channel_configs[channel] = config
        logger.info(f"Updated config for {channel.value}")
    
    def add_websocket_connection(self, websocket) -> None:
        """
        Add WebSocket connection for real-time alerts
        """
        self.websocket_connections.append(websocket)
        logger.debug(f"Added WebSocket connection, total: {len(self.websocket_connections)}")
    
    def remove_websocket_connection(self, websocket) -> None:
        """
        Remove WebSocket connection
        """
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)
            logger.debug(f"Removed WebSocket connection, remaining: {len(self.websocket_connections)}")
    
    async def get_alert_history(
        self,
        alert_type: Optional[AlertType] = None,
        priority: Optional[AlertPriority] = None,
        limit: int = 100
    ) -> List[AlertNotification]:
        """
        Get alert history
        """
        alerts = self.alert_history
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        if priority:
            alerts = [a for a in alerts if a.priority == priority]
        
        return alerts[-limit:]
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert
        """
        for alert in self.alert_history:
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now()
                await self._update_alert_status(alert)
                return True
        return False
    
    async def _on_signal_generated(self, event) -> None:
        """
        Handle signal generated event
        """
        try:
            data = event.data
            signal = data.get("signal")
            agent_id = data.get("agent_id")
            
            if signal:
                await self.send_signal_alert(signal, agent_id)
                
        except Exception as e:
            logger.error(f"Error handling signal event: {str(e)}")
    
    async def _on_risk_alert(self, event) -> None:
        """
        Handle risk alert event
        """
        try:
            data = event.data
            
            await self.send_alert(
                AlertType.RISK,
                AlertPriority(data.get("risk_level", "high")),
                data.get("title", "Risk Alert"),
                data.get("message", ""),
                portfolio_id=data.get("portfolio_id"),
                data=data
            )
            
        except Exception as e:
            logger.error(f"Error handling risk alert: {str(e)}")
    
    async def _on_performance_alert(self, event) -> None:
        """
        Handle performance alert event
        """
        try:
            data = event.data
            
            await self.send_alert(
                AlertType.PERFORMANCE,
                AlertPriority.MEDIUM,
                data.get("title", "Performance Alert"),
                data.get("message", ""),
                agent_id=data.get("agent_id"),
                data=data
            )
            
        except Exception as e:
            logger.error(f"Error handling performance alert: {str(e)}")


# Global alert manager instance
alert_manager = AlertManager()