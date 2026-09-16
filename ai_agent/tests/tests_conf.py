"""AgentSettings factory: call-time reads, optional raw override."""

import os
from unittest.mock import patch

from django.core.exceptions import ImproperlyConfigured
from django.test import SimpleTestCase

from ai_agent.conf import (
    get_agent_settings,
    join_en,
    labeled_terms,
    resolve_authenticate_token,
    resolve_eval_module,
    resolve_studio_django_settings_module,
)


class AgentSettingsFactoryTests(SimpleTestCase):
    def test_raw_override_does_not_need_django_ai_agent(self):
        parsed = get_agent_settings(
            raw={
                "SUPERVISOR_MODEL": "openai:raw-supervisor",
                "SUBAGENT_MODEL": "openai:raw-subagent",
                "COMPACTION_ENABLED": True,
            }
        )
        self.assertEqual(parsed.supervisor_model, "openai:raw-supervisor")
        self.assertEqual(parsed.subagent_model, "openai:raw-subagent")
        self.assertTrue(parsed.compaction_enabled)
        self.assertEqual(parsed.compaction_model, "openai:raw-subagent")

    def test_omitted_models_have_no_provider_default(self):
        parsed = get_agent_settings(raw={})
        self.assertEqual(parsed.supervisor_model, "")
        self.assertEqual(parsed.subagent_model, "")
        self.assertEqual(parsed.memory_embeddings, "")
        self.assertEqual(parsed.authenticate_token, "")

    def test_platform_defaults_when_omitted(self):
        parsed = get_agent_settings(raw={"SUPERVISOR_MODEL": "openai:x"})
        self.assertEqual(parsed.platform.platform_name, "this product")
        self.assertEqual(
            parsed.platform.entity_terms, "values, records, or identifiers"
        )
        self.assertEqual(parsed.platform.domains, ())
        self.assertEqual(parsed.platform.markets_out_of_scope, ())
        self.assertEqual(parsed.platform.mutation_terms, ())
        self.assertEqual(parsed.platform.pii_terms, ())
        self.assertEqual(parsed.platform.currency_tokens, ())
        self.assertEqual(parsed.platform.phone_patterns, ())
        self.assertEqual(parsed.platform.extra_pii_patterns, ())
        self.assertEqual(parsed.platform.extra_downgrade_patterns, ())

    def test_platform_overrides(self):
        parsed = get_agent_settings(
            raw={
                "PLATFORM_NAME": "Acme",
                "PLATFORM_DOMAINS": ("billing", "support"),
                "PLATFORM_ENTITY_TERMS": "invoices or ticket IDs",
                "PLATFORM_MARKETS_OUT_OF_SCOPE": "crypto",
                "PLATFORM_MUTATION_TERMS": ("refunds", "cancellations"),
                "PLATFORM_PII_TERMS": "emails, phone numbers",
                "PLATFORM_CURRENCY_TOKENS": ("usd", "eur"),
                "PLATFORM_PHONE_PATTERNS": r"\d{10}",
                "PLATFORM_EXTRA_PII_PATTERNS": (r"\btkt-\d+\b",),
                "PLATFORM_EXTRA_DOWNGRADE_PATTERNS": (r"\binvent invoices\b",),
            }
        )
        platform = parsed.platform
        self.assertEqual(platform.platform_name, "Acme")
        self.assertEqual(platform.domains, ("billing", "support"))
        self.assertEqual(platform.entity_terms, "invoices or ticket IDs")
        self.assertEqual(platform.markets_out_of_scope, ("crypto",))
        self.assertEqual(platform.mutation_terms, ("refunds", "cancellations"))
        self.assertEqual(platform.pii_terms, ("emails", "phone numbers"))
        self.assertEqual(platform.currency_tokens, ("usd", "eur"))
        self.assertEqual(platform.phone_patterns, (r"\d{10}",))
        self.assertEqual(platform.extra_pii_patterns, (r"\btkt-\d+\b",))
        self.assertEqual(
            platform.extra_downgrade_patterns, (r"\binvent invoices\b",)
        )

    def test_authenticate_token_callable(self):
        def _auth(token):
            return token

        parsed = get_agent_settings(raw={"AUTHENTICATE_TOKEN": _auth})
        self.assertIs(parsed.authenticate_token, _auth)
        self.assertIs(resolve_authenticate_token(raw={"AUTHENTICATE_TOKEN": _auth}), _auth)

    def test_authenticate_token_required(self):
        with self.assertRaises(ImproperlyConfigured):
            resolve_authenticate_token(raw={})

    def test_eval_module_and_trusted_proxy_defaults(self):
        parsed = get_agent_settings(raw={})
        self.assertEqual(parsed.eval_module, "")
        self.assertEqual(parsed.http_trusted_proxy_count, 0)
        self.assertIsNone(resolve_eval_module(raw={}))

    def test_eval_module_and_trusted_proxy_overrides(self):
        parsed = get_agent_settings(
            raw={
                "EVAL_MODULE": "dummy.evals",
                "HTTP_TRUSTED_PROXY_COUNT": 2,
            }
        )
        self.assertEqual(parsed.eval_module, "dummy.evals")
        self.assertEqual(parsed.http_trusted_proxy_count, 2)
        self.assertEqual(resolve_eval_module(raw={"EVAL_MODULE": "dummy.evals"}).__name__, "dummy.evals")


class StudioDjangoSettingsTests(SimpleTestCase):
    def test_prefers_studio_override(self):
        with patch.dict(
            os.environ,
            {
                "DJANGO_SETTINGS_MODULE": "already.set",
                "AI_AGENT_STUDIO_DJANGO_SETTINGS": "host.studio",
            },
            clear=True,
        ):
            self.assertEqual(
                resolve_studio_django_settings_module(), "host.studio"
            )

    def test_keeps_existing_django_settings_module(self):
        with patch.dict(
            os.environ, {"DJANGO_SETTINGS_MODULE": "already.set"}, clear=True
        ):
            self.assertEqual(
                resolve_studio_django_settings_module(), "already.set"
            )

    def test_requires_one_of_the_env_vars(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as caught:
                resolve_studio_django_settings_module()
        self.assertIn("DJANGO_SETTINGS_MODULE", str(caught.exception))


class JoinEnTests(SimpleTestCase):
    def test_join_en_forms(self):
        self.assertEqual(join_en(()), "")
        self.assertEqual(join_en(("wallet",)), "wallet")
        self.assertEqual(join_en(("wallet", "trading")), "wallet and trading")
        self.assertEqual(
            join_en(("wallet", "trading", "funds")),
            "wallet, trading, and funds",
        )

    def test_labeled_terms_omits_empty_parenthetical(self):
        self.assertEqual(
            labeled_terms("out-of-scope topics", ()), "out-of-scope topics"
        )
        self.assertEqual(
            labeled_terms("out-of-scope topics", ("Bitcoin",)),
            "out-of-scope topics (Bitcoin)",
        )
