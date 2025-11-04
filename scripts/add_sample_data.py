"""Add sample documents for testing."""

import asyncio
from tools.api_client import APIGatewayClient


async def main() -> None:
    async with APIGatewayClient() as client:
        # Create collection
        print("Creating collection...")
        try:
            await client.create_collection(
                name="intramind_documents",
                description="IntraMind sample documents",
            )
            print("OK: Collection created")
        except Exception as e:
            print(f"Collection may already exist: {e}")

        # Sample documents
        documents = [
            {
                "content": (
                    "Q4 2024 revenue projections show 25% year-over-year growth. "
                    "Key drivers include new product launches and market expansion in Asia."
                ),
                "metadata": {
                    "title": "Q4 Revenue Projections",
                    "type": "financial",
                    "date": "2024-10-15",
                    "department": "Finance",
                },
            },
            {
                "content": (
                    "The new AI-powered analytics platform is scheduled for release in "
                    "December 2024. Features include real-time data visualization, "
                    "predictive modeling, and automated reporting."
                ),
                "metadata": {
                    "title": "Product Launch Plan",
                    "type": "product",
                    "date": "2024-09-20",
                    "department": "Product",
                },
            },
            {
                "content": (
                    "Annual security audit completed successfully. All systems passed "
                    "compliance checks. Recommendations include implementing MFA for "
                    "all admin accounts and updating SSL certificates quarterly."
                ),
                "metadata": {
                    "title": "Security Audit Report",
                    "type": "security",
                    "date": "2024-10-01",
                    "department": "IT Security",
                },
            },
            {
                "content": (
                    "Customer satisfaction scores increased to 92% in Q3, up from 87% "
                    "in Q2. Primary factors: improved response times and expanded "
                    "self-service options."
                ),
                "metadata": {
                    "title": "Q3 Customer Satisfaction",
                    "type": "customer",
                    "date": "2024-10-10",
                    "department": "Customer Success",
                },
            },
            {
                "content": (
                    "Remote work policy update: Hybrid model will be permanent. "
                    "Employees required in office Tuesday-Thursday. Monday and Friday "
                    "flexible. Home office stipend increased to $500 annually."
                ),
                "metadata": {
                    "title": "Remote Work Policy Update",
                    "type": "hr",
                    "date": "2024-10-05",
                    "department": "Human Resources",
                },
            },
        ]

        # Insert documents
        print(f"\nInserting {len(documents)} documents...")
        for i, doc in enumerate(documents, 1):
            try:
                result = await client.insert_document(
                    collection_name="intramind_documents",
                    content=doc["content"],
                    metadata=doc["metadata"],
                )
                print(f"  {i}. OK: {doc['metadata']['title']} (ID: {result.id})")
            except Exception as e:
                print(f"  {i}. Failed: {e}")

        print("\nSample data loaded!")
        print("\nTry these queries:")
        print("  - 'What are our revenue projections?'")
        print("  - 'Tell me about the product launch'")
        print("  - 'What's the remote work policy?'")


if __name__ == "__main__":
    asyncio.run(main())


