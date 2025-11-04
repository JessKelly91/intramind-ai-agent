"""Demo script showing metrics functionality.

This script demonstrates how metrics are automatically tracked
and can be accessed programmatically.
"""

import asyncio

from agent import IntraMindAgent
from utils.metrics import get_metrics, reset_metrics


async def main():
    """Run metrics demonstration."""
    print("🧠 IntraMind Metrics Demo\n")
    print("=" * 60)
    
    # Reset metrics to start fresh
    print("\n1. Resetting metrics...")
    reset_metrics()
    
    # Create agent
    agent = IntraMindAgent()
    
    # Run some sample searches
    print("\n2. Running sample searches...")
    
    queries = [
        "What is machine learning?",  # Simple query
        "Compare supervised and unsupervised learning algorithms",  # Complex query
        "neural networks",  # Simple query
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"   [{i}/3] Searching: {query[:40]}...")
        try:
            result = await agent.search(
                query,
                collection_name="intramind_documents",
                num_results=5,
            )
            status = "✓" if result.get("success") else "✗"
            complexity = result.get("complexity", "unknown")
            print(f"        {status} {complexity} query")
        except Exception as e:
            print(f"        ✗ Error: {e}")
    
    # Get and display metrics
    print("\n3. Fetching metrics...")
    metrics = get_metrics()
    
    print("\n" + "=" * 60)
    print("📊 METRICS SUMMARY")
    print("=" * 60)
    
    print("\n🔍 Query Metrics:")
    print(f"   Total Queries:    {metrics['queries']['total']}")
    print(f"   Simple Queries:   {metrics['queries']['simple']} ({metrics['queries']['simple_pct']:.1f}%)")
    print(f"   Complex Queries:  {metrics['queries']['complex']} ({metrics['queries']['complex_pct']:.1f}%)")
    
    print("\n⚡ Performance Metrics:")
    print(f"   Average Latency:  {metrics['performance']['avg_latency_s']:.2f}s")
    print(f"   Error Count:      {metrics['errors']['total']}")
    print(f"   Error Rate:       {metrics['errors']['rate']:.1f}%")
    
    print("\n💰 Cost Metrics (Estimated):")
    print(f"   Router Calls:     {metrics['costs']['router_calls']}")
    print(f"   Primary Calls:    {metrics['costs']['primary_calls']}")
    print(f"   Total Cost:       ${metrics['costs']['total_cost_usd']:.4f}")
    print(f"   Cost per Query:   ${metrics['costs']['total_cost_usd'] / metrics['queries']['total']:.6f}")
    
    print("\n🕐 System Info:")
    print(f"   Uptime:           {metrics['system']['uptime']}")
    print(f"   Started:          {metrics['system']['start_time']}")
    
    print("\n" + "=" * 60)
    print("\n💡 Try running 'intramind metrics' to see this in a nice table format!")
    print()


if __name__ == "__main__":
    asyncio.run(main())

