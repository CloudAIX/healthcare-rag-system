# Azure twin — minimal infrastructure for the healthcare RAG retrieval layer.
#
#   az login
#   cd infra/azure
#   terraform init && terraform apply
#   export AZURE_SEARCH_ENDPOINT=$(terraform output -raw search_endpoint)
#   export AZURE_SEARCH_KEY=$(terraform output -raw search_admin_key)
#   python scripts/ingest_azure.py
#
# Cost note: the free tier (one per subscription) carries this corpus easily
# (95 chunks); "basic" is the smallest paid tier if free is taken.

terraform {
  required_version = ">= 1.5"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
}

provider "azurerm" {
  features {}
}

variable "location" {
  description = "Azure region. australiaeast keeps health data onshore."
  type        = string
  default     = "australiaeast"
}

variable "search_sku" {
  description = "Azure AI Search tier: free | basic | standard"
  type        = string
  default     = "free"
}

resource "azurerm_resource_group" "rag" {
  name     = "rg-healthcare-rag"
  location = var.location
}

resource "azurerm_search_service" "rag" {
  name                = "srch-healthcare-rag"
  resource_group_name = azurerm_resource_group.rag.name
  location            = azurerm_resource_group.rag.location
  sku                 = var.search_sku

  local_authentication_enabled = true
}

output "search_endpoint" {
  value = "https://${azurerm_search_service.rag.name}.search.windows.net"
}

output "search_admin_key" {
  value     = azurerm_search_service.rag.primary_key
  sensitive = true
}
