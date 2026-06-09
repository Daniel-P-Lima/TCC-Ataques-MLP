file(GLOB EMBED_FILES_LIST "binTrees/*")

idf_component_register(
    SRCS "app_main.c"
    PRIV_REQUIRES esp-tls esp_wifi esp_netif esp_event nvs_flash
    INCLUDE_DIRS "include"
    EMBED_FILES ${EMBED_FILES_LIST}
)
